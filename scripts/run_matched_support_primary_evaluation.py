#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.r2bda_net import R2BDANet

spec = importlib.util.spec_from_file_location("building_dataset", REPO_ROOT / "scripts" / "10_building_dataset.py")
b_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b_module)
BuildingInstanceDataset = b_module.BuildingInstanceDataset

DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "results" / "models"
DEFAULT_THRESHOLD_FILE = REPO_ROOT / "results" / "unified_global_threshold.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "primary_evaluation_summary.json"

CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]
MODELS = ["GP", "FC", "FI"]


def evaluate_single_checkpoint(
    model_name: str, city: str, seed: int, ckpt_path: Path, tau: float, device: torch.device
) -> dict[str, object]:
    eval_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    base_meta = DATA_ROOT / "09_metadata" / "turkey_worldview_loco_v3_base.parquet"
    manifest_path = DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    target_test_bids = manifest["target_buildings"]["test"]

    use_mask = model_name != "GP"
    mask_mode = "global_matched_14x14" if model_name == "GP" else "matched_support_14x14"
    mask_ablation = "FC" if model_name == "FC" else None

    test_ds = BuildingInstanceDataset(
        metadata_path=base_meta,
        base_dir=DATA_ROOT,
        split="test",
        sensors=["turkey_wv_visual_rgb"],
        transform=eval_transform,
        use_mask=use_mask,
        mask_ablation=mask_ablation,
        mask_mode=mask_mode,
        building_ids=target_test_bids,
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

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

    all_y2 = []
    all_p2 = []
    bids = []

    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device) if use_mask else None
            l2 = batch["label_2class"].numpy()
            bids.extend(batch["building_id"])

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = net(img, "turkey_wv_visual_rgb", mask=mask)
                probs2 = torch.softmax(out["logits_2class"], dim=1)[:, 1].cpu().numpy()

            all_y2.extend(l2)
            all_p2.extend(probs2)

    y_true = np.asarray(all_y2)
    y_prob = np.asarray(all_p2)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    bids = [b for b, v in zip(bids, valid) if v]

    y_pred = (y_prob >= tau).astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_d = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "model": model_name,
        "city": city,
        "seed": seed,
        "n_samples": len(y_true),
        "balanced_accuracy": float(ba),
        "damage_precision": float(prec),
        "damage_recall": float(rec),
        "damaged_f1": float(f1_d),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "positive_rate": float(y_pred.mean()),
        "building_ids": bids,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def compute_checkpoint_first_bootstrap(
    checkpoint_results: dict[str, dict[str, dict[int, dict]]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Execute stratified paired building-level bootstrap for checkpoint-first evaluation."""
    rng = np.random.default_rng(seed)

    city_building_data = {}
    for c in CITIES:
        ref = checkpoint_results["GP"][c][42]
        bids = ref["building_ids"]
        y_true = ref["y_true"]

        pred_dict = {}
        for m in MODELS:
            pred_dict[m] = {}
            for s in SEEDS:
                ckpt = checkpoint_results[m][c][s]
                assert ckpt["building_ids"] == bids, f"Building IDs mismatch in {m} {c} {s}"
                np.testing.assert_array_equal(ckpt["y_true"], y_true, err_msg=f"y_true mismatch in {m} {c} {s}")
                pred_dict[m][s] = ckpt["y_pred"]

        intact_idx = np.where(y_true == 0)[0]
        damaged_idx = np.where(y_true == 1)[0]
        city_building_data[c] = {
            "n_total": len(y_true),
            "y_true": y_true,
            "intact_idx": intact_idx,
            "damaged_idx": damaged_idx,
            "predictions": pred_dict,
        }

    boot_diffs_21 = []
    boot_diffs_10 = []

    for _ in range(n_boot):
        city_model_ba = {m: [] for m in MODELS}

        for c in CITIES:
            c_data = city_building_data[c]
            intact_boot = rng.choice(c_data["intact_idx"], size=len(c_data["intact_idx"]), replace=True)
            damaged_boot = rng.choice(c_data["damaged_idx"], size=len(c_data["damaged_idx"]), replace=True)
            boot_idx = np.concatenate([intact_boot, damaged_boot])
            y_t_boot = c_data["y_true"][boot_idx]

            for m in MODELS:
                seed_bas = []
                for s in SEEDS:
                    y_p_boot = c_data["predictions"][m][s][boot_idx]
                    ba_ckpt = balanced_accuracy_score(y_t_boot, y_p_boot)
                    seed_bas.append(ba_ckpt)
                city_model_ba[m].append(float(np.mean(seed_bas)))

        m0_mean = float(np.mean(city_model_ba["GP"]))
        m1_mean = float(np.mean(city_model_ba["FC"]))
        m2_mean = float(np.mean(city_model_ba["FI"]))

        boot_diffs_21.append(m2_mean - m1_mean)
        boot_diffs_10.append(m1_mean - m0_mean)

    ci_21 = [float(np.percentile(boot_diffs_21, 2.5)), float(np.percentile(boot_diffs_21, 97.5))]
    ci_10 = [float(np.percentile(boot_diffs_10, 2.5)), float(np.percentile(boot_diffs_10, 97.5))]

    return {
        "bootstrap_iterations": n_boot,
        "stratification": "within_city_within_label",
        "delta_21_ci_95": ci_21,
        "delta_10_ci_95": ci_10,
        "delta_21_significant": bool(ci_21[0] > 0 or ci_21[1] < 0),
        "delta_10_significant": bool(ci_10[0] > 0 or ci_10[1] < 0),
    }


def run_primary_evaluation(
    models_dir: Path, threshold_file: Path, output_file: Path, device_name: str = "auto"
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")

    if not threshold_file.exists():
        raise FileNotFoundError(
            f"Unified threshold file not found: {threshold_file}. Run select_unified_global_threshold.py first."
        )
    with open(threshold_file, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    tau_star = float(thresh_info["optimal_threshold_tau_star"])

    print("=" * 80)
    print("PHASE D: PRIMARY COMPARATIVE EVALUATION (TURKEY LOCO)")
    print(f"Protocol: Strict Checkpoint-First Stratified Paired Bootstrap")
    print(f"Unified Global Threshold: tau* = {tau_star:.2f}")
    print(f"Models Directory: {models_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    checkpoint_results = {m: {c: {} for c in CITIES} for m in MODELS}
    raw_eval_serializable = []

    for m in MODELS:
        for c in CITIES:
            for s in SEEDS:
                ckpt_path = models_dir / m / c / f"seed_{s}" / "best.pt"
                print(f"Evaluating checkpoint: {m} | {c} | seed {s}...")
                rec = evaluate_single_checkpoint(m, c, s, ckpt_path, tau_star, device)
                checkpoint_results[m][c][s] = rec
                rec_copy = dict(rec)
                rec_copy.pop("y_true")
                rec_copy.pop("y_prob")
                rec_copy.pop("y_pred")
                raw_eval_serializable.append(rec_copy)

    city_point_means = {m: {} for m in MODELS}
    for m in MODELS:
        for c in CITIES:
            seed_bas = [checkpoint_results[m][c][s]["balanced_accuracy"] for s in SEEDS]
            seed_f1s = [checkpoint_results[m][c][s]["damaged_f1"] for s in SEEDS]
            seed_rocs = [checkpoint_results[m][c][s]["roc_auc"] for s in SEEDS]
            seed_prs = [checkpoint_results[m][c][s]["pr_auc"] for s in SEEDS]
            city_point_means[m][c] = {
                "balanced_accuracy": float(np.mean(seed_bas)),
                "damaged_f1": float(np.mean(seed_f1s)),
                "roc_auc": float(np.mean(seed_rocs)),
                "pr_auc": float(np.mean(seed_prs)),
                "seed_bas": [float(b) for b in seed_bas],
            }

    overall_point_estimates = {}
    for m in MODELS:
        overall_point_estimates[m] = {
            "balanced_accuracy": float(np.mean([city_point_means[m][c]["balanced_accuracy"] for c in CITIES])),
            "damaged_f1": float(np.mean([city_point_means[m][c]["damaged_f1"] for c in CITIES])),
            "roc_auc": float(np.mean([city_point_means[m][c]["roc_auc"] for c in CITIES])),
            "pr_auc": float(np.mean([city_point_means[m][c]["pr_auc"] for c in CITIES])),
        }

    true_delta_21 = overall_point_estimates["FI"]["balanced_accuracy"] - overall_point_estimates["FC"]["balanced_accuracy"]
    true_delta_10 = overall_point_estimates["FC"]["balanced_accuracy"] - overall_point_estimates["GP"]["balanced_accuracy"]

    print("\nRunning 2,000-iteration stratified paired building-level bootstrap...")
    boot_res = compute_checkpoint_first_bootstrap(checkpoint_results, n_boot=2000, seed=42)

    summary = {
        "protocol": "Turkey LOCO Primary Evaluation (Strict Checkpoint-First)",
        "unified_global_threshold_tau_star": tau_star,
        "checkpoint_first_point_estimates": {
            "overall_3city_mean": overall_point_estimates,
            "per_city_mean": city_point_means,
        },
        "primary_contrast_delta_21": {
            "contrast": "FI - FC",
            "point_estimate_pp": float(true_delta_21 * 100),
            "ci_95_pp": [float(boot_res["delta_21_ci_95"][0] * 100), float(boot_res["delta_21_ci_95"][1] * 100)],
            "significant_at_005": boot_res["delta_21_significant"],
        },
        "auxiliary_contrast_delta_10": {
            "contrast": "FC - GP",
            "point_estimate_pp": float(true_delta_10 * 100),
            "ci_95_pp": [float(boot_res["delta_10_ci_95"][0] * 100), float(boot_res["delta_10_ci_95"][1] * 100)],
            "significant_at_005": boot_res["delta_10_significant"],
        },
        "bootstrap_metadata": {
            "iterations": 2000,
            "seed": 42,
            "stratification": "within_city_within_label",
        },
        "raw_checkpoint_evaluations": raw_eval_serializable,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PRIMARY COMPARATIVE EVALUATION SUMMARY (TURKEY LOCO)")
    print("=" * 80)
    for m in MODELS:
        ba = overall_point_estimates[m]["balanced_accuracy"] * 100
        f1 = overall_point_estimates[m]["damaged_f1"] * 100
        print(f"  {m:<10}: BA = {ba:.2f}%, Damaged F1 = {f1:.2f}%")
    print("-" * 80)
    print(f"  Primary Contrast (FI - FC):  {true_delta_21*100:+.2f} pp, 95% CI: [{boot_res['delta_21_ci_95'][0]*100:+.2f}, {boot_res['delta_21_ci_95'][1]*100:+.2f}] pp (Sig: {boot_res['delta_21_significant']})")
    print(f"  Auxiliary Contrast (FC - GP): {true_delta_10*100:+.2f} pp, 95% CI: [{boot_res['delta_10_ci_95'][0]*100:+.2f}, {boot_res['delta_10_ci_95'][1]*100:+.2f}] pp (Sig: {boot_res['delta_10_significant']})")
    print("=" * 80)
    print(f"Saved primary evaluation report to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--threshold-file", type=Path, default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_primary_evaluation(args.models_dir, args.threshold_file, args.output, args.device)


if __name__ == "__main__":
    main()
