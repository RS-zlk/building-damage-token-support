"""Train an R2-BDA model."""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.r2bda_net import R2BDANet
from src.r2_bda.run_provenance import RunProvenance, file_fingerprint

def parse_channel_indices(value: str):
    if not value: return None
    return [int(x.strip()) for x in value.split(',')]

DATA_ROOT = Path(os.environ.get("R2_BDA_DATA_DIR", REPO_ROOT / "data"))



def load_dataset_class():
    spec = importlib.util.spec_from_file_location("building_dataset", REPO_ROOT / "scripts" / "10_building_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BuildingInstanceDataset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import copy


class FocalLoss(nn.Module):
    """Focal loss for binary and auxiliary heads."""
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ModelEMA:
    """Exponential moving average model wrapper."""
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema_model.state_dict()


def class_weights(labels, n_classes, device):
    counts = torch.bincount(labels[labels >= 0], minlength=n_classes).float()
    weights = counts.sum() / (n_classes * counts.clamp_min(1))
    return weights.to(device)


def resolve_input_channels(shape, channel_indices_value):
    """Determine input channels from metadata and optional indices."""
    stored_channels = (
        1
        if len(shape) == 2
        else shape[0]
        if shape[0] <= 16 and shape[-1] > 16
        else shape[-1]
    )
    channel_indices = (
        parse_channel_indices(channel_indices_value)
        if channel_indices_value is not None
        else None
    )
    if channel_indices is not None and max(channel_indices) >= stored_channels:
        raise ValueError(
            f"Channel indices {channel_indices} exceed image channel range [0, {stored_channels - 1}]"
        )
    channels = len(channel_indices) if channel_indices is not None else stored_channels
    return stored_channels, channel_indices, channels


def model_forward(model, batch, sensor, device, use_mask, non_blocking=False):
    mask = batch["mask"].to(device, non_blocking=non_blocking) if use_mask else None
    image = batch["image"].to(device, non_blocking=non_blocking)
    return model(image, sensor, mask=mask)


def dataloader_options(args, device):
    """Build compatible dataloader keyword arguments."""
    pin_memory = args.pin_memory if args.pin_memory is not None else device.type == "cuda"
    persistent_workers = (
        args.persistent_workers
        if args.persistent_workers is not None
        else args.num_workers > 0
    )
    if args.num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    options = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if args.num_workers > 0 else False,
    }
    if args.num_workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
    return options, pin_memory


def training_transform(augmentation_preset: str, *, overfit: bool) -> A.Compose:
    """Build training augmentation."""
    if overfit:
        return A.Compose([A.Resize(224, 224), ToTensorV2()])
    geometric = [A.Resize(224, 224), A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    if augmentation_preset == "standard":
        return A.Compose([*geometric, ToTensorV2()])
    if augmentation_preset == "gf2_source_degradation_v1":
        return A.Compose([
            *geometric,
            A.Downscale(scale_range=(0.5, 0.8), interpolation_pair={"downscale": 3, "upscale": 1}, p=0.85),
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.5, 1.4), p=0.65),
            A.GaussNoise(std_range=(0.015, 0.05), mean_range=(0.0, 0.0), per_channel=True, p=0.45),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.75, 1.25), hue=(-0.04, 0.04), p=0.65),
            ToTensorV2(),
        ])
    if augmentation_preset in ("robust_degradation", "robust_degradation_v1"):
        return A.Compose([
            *geometric,
            A.Downscale(scale_range=(0.25, 0.75), interpolation_pair={"downscale": 3, "upscale": 1}, p=0.85),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.0), p=0.75),
            A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), per_channel=True, p=0.50),
            A.ColorJitter(brightness=(0.75, 1.25), contrast=(0.75, 1.25), saturation=(0.70, 1.30), hue=(-0.05, 0.05), p=0.70),
            ToTensorV2(),
        ])
    raise ValueError(f"Unknown augmentation preset: {augmentation_preset}")


@torch.no_grad()
def evaluate(
    model, loader, sensor, device, loss4, loss2, use_mask=False, amp=False,
    non_blocking=False,
):
    model.eval()
    y4, p4, y2, p2 = [], [], [], []
    total4, total2, n = 0.0, 0.0, 0
    for batch in loader:
        labels4 = batch["label_4class"].to(device, non_blocking=non_blocking)
        labels2 = batch["label_2class"].to(device, non_blocking=non_blocking)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model_forward(
                model, batch, sensor, device, use_mask, non_blocking=non_blocking
            )
            batch_loss4 = loss4(out["logits_4class"], labels4)
            batch_loss2 = loss2(out["logits_2class"], labels2)
        batch_size = labels4.size(0)
        total4 += batch_loss4.item() * batch_size
        total2 += batch_loss2.item() * batch_size
        n += batch_size
        y4.extend(labels4.cpu().tolist())
        p4.extend(out["logits_4class"].argmax(1).cpu().tolist())
        y2.extend(labels2.cpu().tolist())
        p2.extend(out["logits_2class"].argmax(1).cpu().tolist())
    valid4 = np.asarray(y4) >= 0
    valid2 = np.asarray(y2) >= 0
    y4, p4 = np.asarray(y4)[valid4], np.asarray(p4)[valid4]
    y2, p2 = np.asarray(y2)[valid2], np.asarray(p2)[valid2]
    per_class4 = f1_score(y4, p4, labels=range(4), average=None, zero_division=0)
    per_class2 = f1_score(y2, p2, labels=range(2), average=None, zero_division=0)
    binary_ba = float(balanced_accuracy_score(y2, p2)) if len(y2) > 0 else 0.0
    damage_prec = float(precision_score(y2, p2, pos_label=1, zero_division=0)) if len(y2) > 0 else 0.0
    damage_rec = float(recall_score(y2, p2, pos_label=1, zero_division=0)) if len(y2) > 0 else 0.0
    damaged_f1 = float(f1_score(y2, p2, pos_label=1, zero_division=0)) if len(y2) > 0 else 0.0
    return {
        "val_loss_4class": total4 / n if n > 0 else 0.0,
        "val_loss_2class": total2 / n if n > 0 else 0.0,
        "binary_balanced_accuracy": binary_ba,
        "macro_f1_4class": float(f1_score(y4, p4, labels=range(4), average="macro", zero_division=0)),
        "macro_f1_2class": float(f1_score(y2, p2, labels=range(2), average="macro", zero_division=0)),
        "damaged_f1": damaged_f1,
        "damage_precision": damage_prec,
        "damage_recall": damage_rec,
        **{f"f1_4class_c{i}": float(value) for i, value in enumerate(per_class4)},
        **{f"f1_2class_c{i}": float(value) for i, value in enumerate(per_class2)},
        **{f"pred_4class_c{i}": int((p4 == i).sum()) for i in range(4)},
        **{f"pred_2class_c{i}": int((p2 == i).sum()) for i in range(2)},
        "confusion_4class": json.dumps(confusion_matrix(y4, p4, labels=range(4)).tolist()),
        "confusion_2class": json.dumps(confusion_matrix(y2, p2, labels=range(2)).tolist()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DATA_ROOT / "09_metadata/model_ready_patch_index.parquet")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--sensor", default="turkey_wv_visual_rgb")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Path to JSON split manifest generated in P0.")
    parser.add_argument("--split-mode", choices=("source", "target"), default="source",
                        help="Which side of the JSON manifest to use.")
    parser.add_argument(
        "--channel-indices",
        default=None,
        help="Zero-based channel indices selected before normalization.",
    )
    parser.add_argument(
        "--normalization",
        choices=("global_percentile_1_99_v1", "per_channel_percentile_1_99_v1"),
        default="global_percentile_1_99_v1",
        help="Normalization scheme.",
    )
    parser.add_argument("--backbone", choices=("conv", "swin_tiny"), default="conv")
    parser.add_argument("--pretrained-backbone", action="store_true",
                        help="Use ImageNet pretrained Swin-Tiny; the sensor adapter maps selected input channels to RGB features.")
    parser.add_argument("--use-mask", action="store_true",
                        help="Legacy alias for --mask-mode input_concat.")
    parser.add_argument(
        "--mask-mode",
        choices=(
            "none", "input_concat", "masked_rgb", "late_fusion", "dual_region",
            "matched_support_14x14", "global_matched_14x14",
        ),
        default="none",
        help="Footprint-mask use mode.",
    )
    parser.add_argument(
        "--mask-ablation",
        choices=("FC", "FI"),
        default=None,
        help="Support construction rule for matched-support training.",
    )
    parser.add_argument("--ring-width", type=int, default=16,
                        help="Outer-ring width in input pixels for dual_region.")
    parser.add_argument(
        "--augmentation-preset",
        choices=("standard", "gf2_source_degradation_v1", "robust_degradation", "robust_degradation_v1"),
        default="standard",
        help="Training-set augmentation preset.",
    )
    parser.add_argument(
        "--loss-type",
        choices=("ce", "focal"),
        default="ce",
        help="Loss type: ce or focal.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal-loss gamma.")
    parser.add_argument("--damage-weight", type=float, default=1.0,
                        help="Damaged-class loss-weight multiplier.")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable model exponential moving average.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Model EMA decay.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None,
                        help="Pinned host memory; defaults to enabled on CUDA.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep workers alive between epochs; defaults to enabled when workers > 0.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches prefetched by each worker when num-workers > 0.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable automatic mixed precision; currently supported on CUDA.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay multiplier; defaults to 1e-4 to match historical C4b.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-samples-per-split", type=int, default=None,
                        help="Debug-only deterministic cap; never use for reported experiments.")
    parser.add_argument("--overfit-samples", type=int, default=None,
                        help="Debug-only: train and validate on the same N training samples without augmentation.")
    parser.add_argument("--balance-mode", choices=("sampler", "loss", "both", "none"), default="sampler",
                        help="Class balancing strategy. 'both' reproduces the old double-balancing behavior.")
    parser.add_argument(
        "--selection-metric",
        choices=("binary_balanced_accuracy", "macro_f1_4class", "macro_f1_2class"),
        default="binary_balanced_accuracy",
        help="Validation metric used for checkpoint selection.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/worldview_baseline")
    parser.add_argument("--allow-existing-output", action="store_true",
                        help="Allow writing to a non-empty output directory.")
    args = parser.parse_args()
    if args.max_samples_per_split and args.overfit_samples:
        parser.error("--max-samples-per-split and --overfit-samples cannot be used together")
    if args.pretrained_backbone and args.backbone != "swin_tiny":
        parser.error("--pretrained-backbone requires --backbone swin_tiny")
    if args.use_mask:
        if args.mask_mode != "none":
            parser.error("--use-mask cannot be combined with --mask-mode")
        args.mask_mode = "input_concat"
    use_mask = args.mask_mode not in ("none", "global_matched_14x14")
    tracker = RunProvenance(
        args.output_dir,
        args,
        REPO_ROOT,
        "training",
        {"metadata": file_fingerprint(args.metadata), "data_root": args.data_root},
        allow_existing=args.allow_existing_output,
    )
    seed_everything(args.seed)
    detected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(detected if args.device == "auto" else args.device)
    if args.amp and device.type != "cuda":
        parser.error("--amp currently requires a CUDA device")
    loader_options, pin_memory = dataloader_options(args, device)
    non_blocking = pin_memory and device.type == "cuda"

    index = pd.read_parquet(args.metadata)
    available = index[index.sensor.eq(args.sensor)]
    if available.empty:
        raise ValueError(f"No records for sensor={args.sensor}. Available: {sorted(index.sensor.unique())}")
    shape = __import__("tifffile").imread(args.data_root / available.iloc[0].image_path).shape
    stored_channels, channel_indices, channels = resolve_input_channels(
        shape, args.channel_indices
    )

    Dataset = load_dataset_class()
    eval_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    train_transform = training_transform(args.augmentation_preset, overfit=bool(args.overfit_samples))

    if args.split_manifest:
        with open(args.split_manifest, 'r') as f:
            manifest = json.load(f)
        b_ids = manifest[f"{args.split_mode}_buildings"]
    else:
        b_ids = {"train": None, "val": None, "test": None}

    datasets = {split: Dataset(args.metadata, args.data_root, split, args.sensor,
                               transform=train_transform if split == "train" else eval_transform,
                               use_mask=use_mask, mask_ablation=args.mask_ablation,
                               channel_indices=channel_indices,
                               normalization=args.normalization,
                               building_ids=b_ids[split],
                               mask_mode=args.mask_mode)
                for split in ("train", "val", "test")}
    if args.max_samples_per_split:
        for dataset in datasets.values():
            dataset.records = dataset.records[:args.max_samples_per_split]
    if args.overfit_samples:
        by_class = {label: [] for label in range(1, 5)}
        for record in datasets["train"].records:
            by_class[int(record["label_4class"])].append(record)
        records = []
        while len(records) < args.overfit_samples and any(by_class.values()):
            for label in range(1, 5):
                if by_class[label] and len(records) < args.overfit_samples:
                    records.append(by_class[label].pop(0))
        datasets["train"].records = records
        datasets["val"].records = list(records)
    train_labels = torch.tensor([r["label_4class"] - 1 for r in datasets["train"].records], dtype=torch.long)
    binary_labels = torch.tensor([r["label_2class"] for r in datasets["train"].records], dtype=torch.long)
    weights4 = class_weights(train_labels, 4, device)
    weights2 = class_weights(binary_labels, 2, device)
    sample_weights = weights4[train_labels].cpu().double()
    sampler = (WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
               if args.balance_mode in ("sampler", "both") else None)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler,
                            shuffle=sampler is None, **loader_options),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False,
                          **loader_options),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False,
                           **loader_options),
    }
    adapter_dim = 3 if args.pretrained_backbone else 64
    model = R2BDANet(
        {args.sensor: channels},
        adapter_dim=adapter_dim,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        mask_mode=args.mask_mode,
        ring_width=args.ring_width,
    ).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss4_weight = weights4 if args.balance_mode in ("loss", "both") else None
    if args.balance_mode == "sampler":
        probabilities = sample_weights / sample_weights.sum()
        binary_mass = torch.stack([
            probabilities[binary_labels == label].sum() for label in range(2)
        ])
        loss2_weight = (1.0 / (2 * binary_mass.clamp_min(1e-12))).to(device=device, dtype=torch.float32)
    elif args.balance_mode in ("loss", "both"):
        loss2_weight = weights2
    else:
        loss2_weight = None

    if args.damage_weight != 1.0:
        if loss2_weight is None:
            loss2_weight = torch.tensor([1.0, float(args.damage_weight)], device=device, dtype=torch.float32)
        else:
            loss2_weight = loss2_weight.clone()
            loss2_weight[1] *= float(args.damage_weight)

    if args.loss_type == "focal":
        loss4 = FocalLoss(weight=loss4_weight, gamma=args.focal_gamma)
        loss2 = FocalLoss(weight=loss2_weight, gamma=args.focal_gamma)
    else:
        loss4 = nn.CrossEntropyLoss(weight=loss4_weight)
        loss2 = nn.CrossEntropyLoss(weight=loss2_weight)

    history, best = [], -1.0

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        model.train()
        total4, total2, n = 0.0, 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            labels4 = batch["label_4class"].to(device, non_blocking=non_blocking)
            labels2 = batch["label_2class"].to(device, non_blocking=non_blocking)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                out = model_forward(
                    model, batch, args.sensor, device, use_mask,
                    non_blocking=non_blocking,
                )
                batch_loss4 = loss4(out["logits_4class"], labels4)
                batch_loss2 = loss2(out["logits_2class"], labels2)
                loss = batch_loss4 + batch_loss2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            loss_item = loss.item()
            batch_size = labels4.size(0)
            total4 += batch_loss4.item() * batch_size
            total2 += batch_loss2.item() * batch_size
            n += batch_size
            pbar.set_postfix(loss=f"{loss_item:.4f}")
        train_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        eval_model = ema.ema_model if ema is not None else model
        metrics = evaluate(
            eval_model, loaders["val"], args.sensor, device, loss4, loss2, use_mask,
            amp=args.amp, non_blocking=non_blocking,
        )
        metrics.update(
            epoch=epoch,
            train_loss_4class=total4 / n,
            train_loss_2class=total2 / n,
            train_loss=(total4 + total2) / n,
            train_seconds=train_seconds,
            train_samples_per_second=n / train_seconds,
            epoch_seconds=time.perf_counter() - epoch_started,
            peak_cuda_memory_gb=(
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda" else 0.0
            ),
            lr=opt.param_groups[0]["lr"],
        )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if metrics[args.selection_metric] > best:
            best = metrics[args.selection_metric]
            saved_state_dict = eval_model.state_dict()
            torch.save({"model": saved_state_dict, "sensor": args.sensor, "channels": channels,
                        "stored_channels": stored_channels,
                        "channel_indices": list(channel_indices) if channel_indices is not None else None,
                        "normalization": args.normalization,
                        "backbone": args.backbone, "adapter_dim": adapter_dim,
                        "pretrained_backbone": args.pretrained_backbone,
                        "use_mask": use_mask,
                        "mask_mode": args.mask_mode,
                        "ring_width": args.ring_width,
                        "selection_metric": args.selection_metric,
                        "use_ema": args.use_ema,
                        "ema_decay": args.ema_decay if args.use_ema else None,
                        "loss_type": args.loss_type,
                        "damage_weight": args.damage_weight,
                        "augmentation_preset": args.augmentation_preset,
                        "epoch": epoch,
                        "batch_size": args.batch_size, "num_workers": args.num_workers,
                        "pin_memory": pin_memory,
                        "persistent_workers": loader_options["persistent_workers"],
                        "prefetch_factor": args.prefetch_factor,
                        "amp": args.amp,
                        "val_metrics": metrics}, args.output_dir / "best.pt")
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_row = max(history, key=lambda row: row[args.selection_metric])
    tracker.complete(
        requested_epochs=args.epochs,
        completed_epochs=len(history),
        best_epoch=best_row["epoch"],
        selection_metric=args.selection_metric,
        best_validation_score=best,
        checkpoint=args.output_dir / "best.pt",
    )
    print(
        f"Saved best checkpoint ({args.selection_metric}={best:.4f}) "
        f"to {args.output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
