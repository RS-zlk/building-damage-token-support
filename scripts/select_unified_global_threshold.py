#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.r2bda_net import R2BDANet
import importlib.util

spec = importlib.util.spec_from_file_location("building_dataset", REPO_ROOT / "scripts" / "10_building_dataset.py")
b_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b_module)
BuildingInstanceDataset = b_module.BuildingInstanceDataset

DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "results" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "unified_global_threshold.json"

CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]
MODELS = ["GP", "FC", "FI"]


def load_val_predictions_for_checkpoint(
    model_name: str, city: str, seed: int, ckpt_path: Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on the source validation set for a specific checkpoint."""
    meta_file = DATA_ROOT / "09_metadata" / f"stage1_source_only_target_{city}.parquet"
    manifest_file = DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    val_bids = manifest["source_buildings"]["val"]

    eval_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    use_mask = model_name != "GP"
    mask_mode = "global_matched_14x14" if model_name == "GP" else "matched_support_14x14"
    mask_ablation = "FC" if model_name == "FC" else None

    val_ds = BuildingInstanceDataset(
        metadata_path=meta_file,
        base_dir=DATA_ROOT,
        split="val",
        sensors=["turkey_wv_visual_rgb"],
        transform=eval_transform,
        use_mask=use_mask,
        mask_ablation=mask_ablation,
        mask_mode=mask_mode,
        building_ids=val_bids,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = R2BDANet(
        {"turkey_wv_visual_rgb": 3},
        adapter_dim=3,
        backbone_name="swin_tiny",
        pretrained_backbone=False,
        mask_mode=mask_mode,
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    all_y_true = []
    all_y_prob = []

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device) if use_mask else None
            labels2 = batch["label_2class"].numpy()

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = net(img, "turkey_wv_visual_rgb", mask=mask)
                probs = torch.softmax(out["logits_2class"], dim=1)[:, 1].cpu().numpy()

            all_y_true.extend(labels2)
            all_y_prob.extend(probs)

    y_true = np.asarray(all_y_true)
    y_prob = np.asarray(all_y_prob)
    valid = y_true >= 0
    return y_true[valid], y_prob[valid]


def select_best_threshold(
    tau_scores: list[dict[str, float]],
    target_center: float = 0.50,
    tol: float = 1e-12,
) -> tuple[dict[str, float], list[str]]:
    if not tau_scores:
        raise ValueError("tau_scores cannot be empty")

    max_ba = max(e["mean_source_val_ba"] for e in tau_scores)
    tied = [e for e in tau_scores if max_ba - e["mean_source_val_ba"] <= tol]
    log_reasons = []

    if len(tied) == 1:
        winner = tied[0]
        log_reasons.append(
            f"Rule 1 (Unique Maximum BA): tau = {winner['tau']:.2f} with BA = {winner['mean_source_val_ba']:.6f}."
        )
        return winner, log_reasons

    log_reasons.append(
        f"Score tie detected among {len(tied)} candidates with BA = {max_ba:.6f}: {[e['tau'] for e in tied]}."
    )

    min_dist = min(abs(e["tau"] - target_center) for e in tied)
    tied_dist = [e for e in tied if abs(abs(e["tau"] - target_center) - min_dist) <= 1e-9]

    if len(tied_dist) == 1:
        winner = tied_dist[0]
        log_reasons.append(
            f"Rule 2 (Closest to {target_center:.2f}): tau = {winner['tau']:.2f} (distance = {min_dist:.4f})."
        )
        return winner, log_reasons

    log_reasons.append(
        f"Distance tie detected among {[e['tau'] for e in tied_dist]} (distance = {min_dist:.4f})."
    )

    winner = min(tied_dist, key=lambda e: e["tau"])
    log_reasons.append(
        f"Rule 3 (Smaller tau fallback): selected smaller tau = {winner['tau']:.2f}."
    )
    return winner, log_reasons


def find_unified_global_threshold(models_dir: Path, output_file: Path, device_name: str = "auto") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print("=" * 80)
    print("PHASE D: UNIFIED GLOBAL THRESHOLD SELECTION")
    print(f"Models Directory: {models_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    checkpoint_val_data = []
    missing_ckpts = []

    for m in MODELS:
        for c in CITIES:
            for s in SEEDS:
                ckpt_path = models_dir / m / c / f"seed_{s}" / "best.pt"
                if not ckpt_path.exists():
                    missing_ckpts.append(str(ckpt_path))
                    continue
                print(f"Collecting validation predictions: {m} | {c} | seed {s}...")
                y_t, y_p = load_val_predictions_for_checkpoint(m, c, s, ckpt_path, device)
                checkpoint_val_data.append({"model": m, "city": c, "seed": s, "y_true": y_t, "y_prob": y_p})

    if missing_ckpts:
        raise FileNotFoundError(
            f"Missing {len(missing_ckpts)} checkpoints out of 27! Ensure training is complete.\n"
            f"Example: {missing_ckpts[0]}"
        )

    print(f"\nAll 27 checkpoints validation data collected successfully ({len(checkpoint_val_data)} sets).")
    print("Searching for optimal global threshold tau* in [0.10, 0.90] in steps of 0.01...")

    taus = np.arange(0.10, 0.91, 0.01)
    tau_scores = []

    for tau in taus:
        scores = []
        for ckpt_data in checkpoint_val_data:
            y_t = ckpt_data["y_true"]
            y_pred = (ckpt_data["y_prob"] >= tau).astype(int)
            ba = balanced_accuracy_score(y_t, y_pred)
            scores.append(ba)
        mean_ba = float(np.mean(scores))
        tau_scores.append({"tau": round(float(tau), 2), "mean_source_val_ba": mean_ba})

    best_entry, tie_logs = select_best_threshold(tau_scores, target_center=0.50, tol=1e-12)
    best_tau = best_entry["tau"]
    best_ba = best_entry["mean_source_val_ba"]

    print("\n" + "=" * 80)
    print(f"OPTIMAL UNIFIED GLOBAL THRESHOLD FOUND: tau* = {best_tau:.2f}")
    print(f"Source-Validation Mean Balanced Accuracy: {best_ba*100:.2f}%")
    print("Decision Rationale:")
    for log in tie_logs:
        print(f"  - {log}")
    print("=" * 80)

    payload = {
        "protocol": "Unified Global Source-Validation Decision Threshold",
        "optimal_threshold_tau_star": best_tau,
        "source_val_mean_ba": best_ba,
        "tie_breaking_rules": [
            "Rule 1: Highest mean source-validation Balanced Accuracy across all 27 checkpoints (tolerance: 1e-12)",
            "Rule 2: If tied within tolerance, select tau closest to 0.50: min |tau - 0.50|",
            "Rule 3: If still tied (equidistant from 0.50, e.g. 0.48 vs 0.52), select the smaller tau",
        ],
        "decision_rationale": tie_logs,
        "search_grid": {
            "min_tau": 0.10,
            "max_tau": 0.90,
            "step": 0.01,
        },
        "n_checkpoints_evaluated": len(checkpoint_val_data),
        "tau_curve": tau_scores,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved threshold specification to: {output_file}\n")
    return best_tau


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    find_unified_global_threshold(args.models_dir, args.output, args.device)


if __name__ == "__main__":
    main()
