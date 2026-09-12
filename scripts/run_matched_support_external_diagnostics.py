#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tifffile
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
DEFAULT_OUTPUT = REPO_ROOT / "results" / "external_diagnostics_summary.json"

MODELS = ["GP", "FC", "FI"]
CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, following checkpoint symlinks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    """Read a Git value without making evaluation depend on Git availability."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def build_evaluation_provenance(
    models_dir: Path, threshold_file: Path, tau_star: float
) -> dict[str, object]:
    """Capture every frozen code and model input required to replay Phase E."""
    checkpoints = []
    for model_name in MODELS:
        for city in CITIES:
            for seed in SEEDS:
                checkpoint_path = models_dir / model_name / city / f"seed_{seed}" / "best.pt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint_path}")
                checkpoints.append(
                    {
                        "model": model_name,
                        "target_city_fold": city,
                        "seed": seed,
                        "path": str(checkpoint_path),
                        "resolved_path": str(checkpoint_path.resolve()),
                        "size_bytes": checkpoint_path.stat().st_size,
                        "sha256": sha256_file(checkpoint_path),
                    }
                )

    selection_manifest = models_dir.parent / "selected_recipe_manifest.json"
    selection_info: dict[str, object] = {"path": str(selection_manifest), "exists": selection_manifest.is_file()}
    if selection_manifest.is_file():
        selection_info["sha256"] = sha256_file(selection_manifest)
        with selection_manifest.open("r", encoding="utf-8") as handle:
            selection_info["selected_recipe_id"] = json.load(handle).get("selected_recipe_id")

    script_path = Path(__file__).resolve()
    git_status = git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": git_value("rev-parse", "HEAD"),
            "status_porcelain": git_status,
            "dirty": git_status not in ("", "UNAVAILABLE"),
        },
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "threshold": {
            "path": str(threshold_file),
            "sha256": sha256_file(threshold_file),
            "tau_star": tau_star,
        },
        "selected_recipe_manifest": selection_info,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
    }


def read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        table = pq.read_table(path, use_threads=False)
        return table.to_pandas()
    return pd.read_csv(path)


def filter_valid_non_empty_masks(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter out any records with missing or empty raw footprint masks."""
    valid_indices = []
    for idx, row in frame.iterrows():
        mask_rel = row.get("mask_path")
        if pd.isna(mask_rel) or not str(mask_rel).strip():
            parts = list(Path(str(row["image_path"])).parts)
            parts[parts.index("images")] = "masks"
            mask_rel = Path(*parts)
        mask_path = DATA_ROOT / mask_rel
        if not mask_path.exists():
            continue
        raw = np.squeeze(tifffile.imread(mask_path))
        if (raw > 0).sum() > 0:
            valid_indices.append(idx)
    return frame.loc[valid_indices].reset_index(drop=True)


def get_cohort_batches(
    df: pd.DataFrame,
    sensor_name: str,
    model_name: str,
    batch_size: int = 64,
) -> list[tuple[torch.Tensor, torch.Tensor | None, np.ndarray, list[str]]]:
    """Pre-load cohort images and masks for a specific model architecture."""
    eval_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    use_mask = model_name != "GP"
    mask_mode = "global_matched_14x14" if model_name == "GP" else "matched_support_14x14"
    mask_ablation = "FC" if model_name == "FC" else None

    temp_dir = REPO_ROOT / "results" / "scratch"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_meta = temp_dir / f"temp_{sensor_name}_{os_pid_rand()}.parquet"
    df.to_parquet(temp_meta, index=False)

    channel_indices = (2, 1, 0) if "gf" in sensor_name.lower() else (0, 1, 2)

    try:
        ds = BuildingInstanceDataset(
            metadata_path=temp_meta,
            base_dir=DATA_ROOT,
            split="test" if "split" in df.columns and "test" in df["split"].values else "val",
            sensors=[sensor_name],
            transform=eval_transform,
            use_mask=use_mask,
            mask_ablation=mask_ablation,
            mask_mode=mask_mode,
            building_ids=df["building_id"].tolist(),
            channel_indices=channel_indices,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        batches = []
        for batch in loader:
            batches.append((
                batch["image"],
                batch["mask"] if use_mask else None,
                batch["label_2class"].numpy(),
                list(batch["building_id"]),
            ))
        return batches
    finally:
        if temp_meta.exists():
            temp_meta.unlink()


def evaluate_cohort_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor | None, np.ndarray, list[str]]],
    sensor_name: str,
    model_name: str,
    ckpt_path: Path,
    tau: float,
    device: torch.device,
) -> dict[str, object]:
    """Run inference on pre-loaded batches with a single checkpoint."""
    use_mask = model_name != "GP"
    mask_mode = "global_matched_14x14" if model_name == "GP" else "matched_support_14x14"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = R2BDANet(
        {sensor_name: 3},
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
        for img, mask, l2, b_ids in batches:
            img = img.to(device)
            mask = mask.to(device) if use_mask else None
            bids.extend(b_ids)

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = net(img, sensor_name, mask=mask)
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


def os_pid_rand() -> str:
    import os
    import time
    return f"{os.getpid()}_{int(time.time()*1000)%100000}"


def run_stratified_bootstrap_cohort(
    checkpoint_preds: dict[str, dict[str, dict[int, np.ndarray]]],
    y_true_dict: dict[str, np.ndarray],
    cohort_type: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    keys = list(y_true_dict.keys())

    boot_model_bas = {m: [] for m in MODELS}
    boot_diffs_21 = []
    boot_diffs_10 = []

    group_indices = {}
    for k in keys:
        yt = y_true_dict[k]
        group_indices[k] = {
            "intact": np.where(yt == 0)[0],
            "damaged": np.where(yt == 1)[0],
            "y_true": yt,
        }

    for _ in range(n_boot):
        model_scores = {m: [] for m in MODELS}

        if cohort_type == "gf1":
            for c in keys:
                g = group_indices[c]
                intact_boot = rng.choice(g["intact"], size=len(g["intact"]), replace=True)
                damaged_boot = rng.choice(g["damaged"], size=len(g["damaged"]), replace=True)
                boot_idx = np.concatenate([intact_boot, damaged_boot])
                y_t_boot = g["y_true"][boot_idx]

                for m in MODELS:
                    seed_bas = [
                        balanced_accuracy_score(y_t_boot, checkpoint_preds[m][c][s][boot_idx])
                        for s in SEEDS
                    ]
                    model_scores[m].append(float(np.mean(seed_bas)))

            m_bas = {m: float(np.mean(model_scores[m])) for m in MODELS}

        elif cohort_type == "single_fold":
            c = keys[0]
            g = group_indices[c]
            intact_boot = rng.choice(g["intact"], size=len(g["intact"]), replace=True)
            damaged_boot = rng.choice(g["damaged"], size=len(g["damaged"]), replace=True)
            boot_idx = np.concatenate([intact_boot, damaged_boot])
            y_t_boot = g["y_true"][boot_idx]

            m_bas = {}
            for m in MODELS:
                seed_bas = [
                    balanced_accuracy_score(y_t_boot, checkpoint_preds[m][c][s][boot_idx])
                    for s in SEEDS
                ]
                m_bas[m] = float(np.mean(seed_bas))

        elif cohort_type == "multi_fold":
            k = keys[0]
            g = group_indices[k]
            intact_boot = rng.choice(g["intact"], size=len(g["intact"]), replace=True)
            damaged_boot = rng.choice(g["damaged"], size=len(g["damaged"]), replace=True)
            boot_idx = np.concatenate([intact_boot, damaged_boot])
            y_t_boot = g["y_true"][boot_idx]

            m_bas = {}
            for m in MODELS:
                fold_bas = []
                for f in keys:
                    seed_bas = [
                        balanced_accuracy_score(y_t_boot, checkpoint_preds[m][f][s][boot_idx])
                        for s in SEEDS
                    ]
                    fold_bas.append(float(np.mean(seed_bas)))
                m_bas[m] = float(np.mean(fold_bas))

        for m in MODELS:
            boot_model_bas[m].append(m_bas[m])
        boot_diffs_21.append(m_bas["FI"] - m_bas["FC"])
        boot_diffs_10.append(m_bas["FC"] - m_bas["GP"])

    ci_dict = {}
    for m in MODELS:
        ci_dict[m] = [
            float(np.percentile(boot_model_bas[m], 2.5)),
            float(np.percentile(boot_model_bas[m], 97.5)),
        ]
    ci_21 = [float(np.percentile(boot_diffs_21, 2.5)), float(np.percentile(boot_diffs_21, 97.5))]
    ci_10 = [float(np.percentile(boot_diffs_10, 2.5)), float(np.percentile(boot_diffs_10, 97.5))]

    return {
        "bootstrap_iterations": n_boot,
        "model_ba_ci_95": ci_dict,
        "delta_21_ci_95": ci_21,
        "delta_10_ci_95": ci_10,
        "delta_21_significant": bool(ci_21[0] > 0 or ci_21[1] < 0),
        "delta_10_significant": bool(ci_10[0] > 0 or ci_10[1] < 0),
    }


def run_external_diagnostics(
    models_dir: Path, threshold_file: Path, output_file: Path, device_name: str = "auto"
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    if not threshold_file.exists():
        raise FileNotFoundError(f"Threshold file {threshold_file} missing. Run threshold selection first.")

    with open(threshold_file, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    tau_star = float(thresh_info["optimal_threshold_tau_star"])

    print("=" * 80)
    print("PHASE E: FROZEN POST-HOC EXTERNAL DIAGNOSTICS (14x14 MATCHED SUPPORT)")
    print(f"Unified Global Threshold: tau* = {tau_star:.2f}")
    print(f"Models Directory: {models_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    print("Loading and filtering cohort metadata...")
    external_df = read_metadata(DATA_ROOT / "09_metadata" / "external_test_patch_index.parquet")
    gf_df = read_metadata(DATA_ROOT / "09_metadata" / "model_ready_patch_index_turkey_gf.parquet")
    venezuela_df = read_metadata(DATA_ROOT / "09_metadata" / "model_ready_patch_index_venezuela_wv.csv")

    haiti_raw = external_df[external_df["dataset"].eq("haiti_wv")].reset_index(drop=True)
    qqb_raw = external_df[external_df["dataset"].eq("qqb_wv")].reset_index(drop=True)
    gf1_raw = gf_df[gf_df["sensor"].eq("turkey_gf1_mtf_glp_cbd")].reset_index(drop=True)
    gf2_raw = gf_df[gf_df["sensor"].eq("turkey_gf2_mtf_glp_cbd")].reset_index(drop=True)
    venezuela_raw = venezuela_df.reset_index(drop=True)

    haiti_clean = filter_valid_non_empty_masks(haiti_raw)
    qqb_clean = filter_valid_non_empty_masks(qqb_raw)
    gf1_clean = filter_valid_non_empty_masks(gf1_raw)
    gf2_clean = filter_valid_non_empty_masks(gf2_raw)
    venezuela_clean = filter_valid_non_empty_masks(venezuela_raw)

    print(f"  Haiti valid non-empty: {len(haiti_clean)} (sensor: {haiti_clean['sensor'].iloc[0]})")
    print(f"  QQB valid non-empty:   {len(qqb_clean)} (sensor: {qqb_clean['sensor'].iloc[0]})")
    print(f"  GF-1 valid non-empty:  {len(gf1_clean)} (sensor: {gf1_clean['sensor'].iloc[0]})")
    print(f"  GF-2 valid non-empty:  {len(gf2_clean)} (sensor: {gf2_clean['sensor'].iloc[0]})")
    print(f"  Venezuela valid non-empty:    {len(venezuela_clean)} (sensor: {venezuela_clean['sensor'].iloc[0]})")

    results = {
        "protocol": "Frozen Post-Hoc External Diagnostics (Strict Checkpoint-First & Target Fold Routing)",
        "unified_global_threshold_tau_star": tau_star,
        "evaluation_provenance": build_evaluation_provenance(models_dir, threshold_file, tau_star),
        "cohorts": {},
    }

    print("\n" + "-" * 80)
    print(f"EVALUATING COHORT: GF1_MTF_historical ({len(gf1_clean)} valid buildings)")
    print("Fold routing rule: Antakya -> Antakya fold; Nurdagi -> Nurdagi fold; Kahramanmaras -> Kahramanmaras fold")
    print("-" * 80)

    gf1_city_dfs = {
        c: gf1_clean[gf1_clean["city"].str.lower() == c].reset_index(drop=True)
        for c in CITIES
    }
    gf1_sensor = gf1_clean["sensor"].iloc[0]

    gf1_preds = {m: {c: {} for c in CITIES} for m in MODELS}
    gf1_raw_evals = []
    gf1_y_trues = {}

    for c in CITIES:
        c_df = gf1_city_dfs[c]
        print(f"  Evaluating GF-1 City Cohort: {c} ({len(c_df)} buildings) using target-city fold: {c}...")
        for m in MODELS:
            batches = get_cohort_batches(c_df, gf1_sensor, m)
            for s in SEEDS:
                ckpt_path = models_dir / m / c / f"seed_{s}" / "best.pt"
                rec = evaluate_cohort_batches(batches, gf1_sensor, m, ckpt_path, tau_star, device)
                gf1_preds[m][c][s] = rec["y_pred"]
                if c not in gf1_y_trues:
                    gf1_y_trues[c] = rec["y_true"]
                rec_copy = dict(rec)
                rec_copy.pop("y_true")
                rec_copy.pop("y_prob")
                rec_copy.pop("y_pred")
                rec_copy["city"] = c
                rec_copy["seed"] = s
                gf1_raw_evals.append(rec_copy)

    gf1_city_point_means = {m: {} for m in MODELS}
    for m in MODELS:
        for c in CITIES:
            bas = [
                balanced_accuracy_score(gf1_y_trues[c], gf1_preds[m][c][s])
                for s in SEEDS
            ]
            gf1_city_point_means[m][c] = {
                "balanced_accuracy": float(np.mean(bas)),
                "seed_bas": [float(b) for b in bas],
            }

    gf1_overall_ba = {
        m: float(np.mean([gf1_city_point_means[m][c]["balanced_accuracy"] for c in CITIES]))
        for m in MODELS
    }
    gf1_delta_21 = gf1_overall_ba["FI"] - gf1_overall_ba["FC"]
    gf1_delta_10 = gf1_overall_ba["FC"] - gf1_overall_ba["GP"]

    gf1_boot = run_stratified_bootstrap_cohort(gf1_preds, gf1_y_trues, cohort_type="gf1", n_boot=2000, seed=42)

    results["cohorts"]["GF1_MTF_historical"] = {
        "n_samples": len(gf1_clean),
        "sensor": gf1_sensor,
        "routing_rule": "Target-city fold matched per cohort: Antakya(3503)->antakya_fold, Nurdagi(2916)->nurdagi_fold, Kahramanmaras(456)->kahramanmaras_fold",
        "city_point_estimates": gf1_city_point_means,
        "checkpoint_first_3city_equal_weight_ba": gf1_overall_ba,
        "primary_contrast_delta_21": {
            "point_estimate_pp": float(gf1_delta_21 * 100),
            "ci_95_pp": [float(gf1_boot["delta_21_ci_95"][0] * 100), float(gf1_boot["delta_21_ci_95"][1] * 100)],
            "significant_at_005": gf1_boot["delta_21_significant"],
        },
        "auxiliary_contrast_delta_10": {
            "point_estimate_pp": float(gf1_delta_10 * 100),
            "ci_95_pp": [float(gf1_boot["delta_10_ci_95"][0] * 100), float(gf1_boot["delta_10_ci_95"][1] * 100)],
            "significant_at_005": gf1_boot["delta_10_significant"],
        },
        "model_ba_ci_95": {m: [float(ci[0]*100), float(ci[1]*100)] for m, ci in gf1_boot["model_ba_ci_95"].items()},
        "raw_evaluations": gf1_raw_evals,
    }

    print("  GF-1 Results:")
    for m in MODELS:
        print(f"    {m:<10}: Overall BA = {gf1_overall_ba[m]*100:.2f}%, 95% CI: [{gf1_boot['model_ba_ci_95'][m][0]*100:.2f}, {gf1_boot['model_ba_ci_95'][m][1]*100:.2f}]%")
    print(f"    Delta_21* (FI - FC): {gf1_delta_21*100:+.2f} pp, 95% CI: [{gf1_boot['delta_21_ci_95'][0]*100:+.2f}, {gf1_boot['delta_21_ci_95'][1]*100:+.2f}] pp (Sig: {gf1_boot['delta_21_significant']})")
    print(f"    Delta_10* (FC - GP): {gf1_delta_10*100:+.2f} pp, 95% CI: [{gf1_boot['delta_10_ci_95'][0]*100:+.2f}, {gf1_boot['delta_10_ci_95'][1]*100:+.2f}] pp (Sig: {gf1_boot['delta_10_significant']})")

    print("\n" + "-" * 80)
    print(f"EVALUATING COHORT: GF2_MTF_historical ({len(gf2_clean)} valid buildings)")
    print("Fold routing rule: Evaluated ONLY with Kahramanmaras target-city fold (3 seeds)")
    print("-" * 80)

    gf2_sensor = gf2_clean["sensor"].iloc[0]
    gf2_preds = {m: {"kahramanmaras": {}} for m in MODELS}
    gf2_raw_evals = []
    gf2_y_true = None

    for m in MODELS:
        batches = get_cohort_batches(gf2_clean, gf2_sensor, m)
        for s in SEEDS:
            ckpt_path = models_dir / m / "kahramanmaras" / f"seed_{s}" / "best.pt"
            rec = evaluate_cohort_batches(batches, gf2_sensor, m, ckpt_path, tau_star, device)
            gf2_preds[m]["kahramanmaras"][s] = rec["y_pred"]
            if gf2_y_true is None:
                gf2_y_true = rec["y_true"]
            rec_copy = dict(rec)
            rec_copy.pop("y_true")
            rec_copy.pop("y_prob")
            rec_copy.pop("y_pred")
            rec_copy["city"] = "kahramanmaras"
            rec_copy["seed"] = s
            gf2_raw_evals.append(rec_copy)

    gf2_point_ba = {}
    for m in MODELS:
        bas = [balanced_accuracy_score(gf2_y_true, gf2_preds[m]["kahramanmaras"][s]) for s in SEEDS]
        gf2_point_ba[m] = float(np.mean(bas))

    gf2_delta_21 = gf2_point_ba["FI"] - gf2_point_ba["FC"]
    gf2_delta_10 = gf2_point_ba["FC"] - gf2_point_ba["GP"]

    gf2_boot = run_stratified_bootstrap_cohort(
        gf2_preds, {"kahramanmaras": gf2_y_true}, cohort_type="single_fold", n_boot=2000, seed=42
    )

    results["cohorts"]["GF2_MTF_historical"] = {
        "n_samples": len(gf2_clean),
        "sensor": gf2_sensor,
        "routing_rule": "Kahramanmaras target-city fold only (3 seeds: 42, 43, 44)",
        "checkpoint_first_seed_average_ba": gf2_point_ba,
        "primary_contrast_delta_21": {
            "point_estimate_pp": float(gf2_delta_21 * 100),
            "ci_95_pp": [float(gf2_boot["delta_21_ci_95"][0] * 100), float(gf2_boot["delta_21_ci_95"][1] * 100)],
            "significant_at_005": gf2_boot["delta_21_significant"],
        },
        "auxiliary_contrast_delta_10": {
            "point_estimate_pp": float(gf2_delta_10 * 100),
            "ci_95_pp": [float(gf2_boot["delta_10_ci_95"][0] * 100), float(gf2_boot["delta_10_ci_95"][1] * 100)],
            "significant_at_005": gf2_boot["delta_10_significant"],
        },
        "model_ba_ci_95": {m: [float(ci[0]*100), float(ci[1]*100)] for m, ci in gf2_boot["model_ba_ci_95"].items()},
        "raw_evaluations": gf2_raw_evals,
    }

    print("  GF-2 Results:")
    for m in MODELS:
        print(f"    {m:<10}: Seed-Avg BA = {gf2_point_ba[m]*100:.2f}%, 95% CI: [{gf2_boot['model_ba_ci_95'][m][0]*100:.2f}, {gf2_boot['model_ba_ci_95'][m][1]*100:.2f}]%")
    print(f"    Delta_21* (FI - FC): {gf2_delta_21*100:+.2f} pp, 95% CI: [{gf2_boot['delta_21_ci_95'][0]*100:+.2f}, {gf2_boot['delta_21_ci_95'][1]*100:+.2f}] pp (Sig: {gf2_boot['delta_21_significant']})")
    print(f"    Delta_10* (FC - GP): {gf2_delta_10*100:+.2f} pp, 95% CI: [{gf2_boot['delta_10_ci_95'][0]*100:+.2f}, {gf2_boot['delta_10_ci_95'][1]*100:+.2f}] pp (Sig: {gf2_boot['delta_10_significant']})")

    multi_fold_cohorts = {
        "Haiti_external": (haiti_clean, haiti_clean["sensor"].iloc[0]),
        "QQB_external": (qqb_clean, qqb_clean["sensor"].iloc[0]),
        "Venezuela_historical_external": (venezuela_clean, venezuela_clean["sensor"].iloc[0]),
    }

    for c_name, (c_df, c_sensor) in multi_fold_cohorts.items():
        print("\n" + "-" * 80)
        print(f"EVALUATING COHORT: {c_name} ({len(c_df)} valid buildings)")
        print(f"Fold routing rule: Evaluated across all 9 checkpoints (3 folds x 3 seeds), sensor: {c_sensor}")
        print("-" * 80)

        preds = {m: {c: {} for c in CITIES} for m in MODELS}
        raw_evals = []
        y_true = None

        for m in MODELS:
            batches = get_cohort_batches(c_df, c_sensor, m)
            for c in CITIES:
                for s in SEEDS:
                    ckpt_path = models_dir / m / c / f"seed_{s}" / "best.pt"
                    rec = evaluate_cohort_batches(batches, c_sensor, m, ckpt_path, tau_star, device)
                    preds[m][c][s] = rec["y_pred"]
                    if y_true is None:
                        y_true = rec["y_true"]
                    rec_copy = dict(rec)
                    rec_copy.pop("y_true")
                    rec_copy.pop("y_prob")
                    rec_copy.pop("y_pred")
                    rec_copy["fold"] = c
                    rec_copy["seed"] = s
                    raw_evals.append(rec_copy)

        fold_point_means = {m: {} for m in MODELS}
        for m in MODELS:
            for c in CITIES:
                bas = [balanced_accuracy_score(y_true, preds[m][c][s]) for s in SEEDS]
                fold_point_means[m][c] = {
                    "balanced_accuracy": float(np.mean(bas)),
                    "seed_bas": [float(b) for b in bas],
                }

        overall_ba = {
            m: float(np.mean([fold_point_means[m][c]["balanced_accuracy"] for c in CITIES]))
            for m in MODELS
        }
        delta_21 = overall_ba["FI"] - overall_ba["FC"]
        delta_10 = overall_ba["FC"] - overall_ba["GP"]

        y_true_dict = {c: y_true for c in CITIES}
        boot = run_stratified_bootstrap_cohort(preds, y_true_dict, cohort_type="multi_fold", n_boot=2000, seed=42)

        results["cohorts"][c_name] = {
            "n_samples": len(c_df),
            "sensor": c_sensor,
            "routing_rule": "All 9 checkpoints (3 folds x 3 seeds); fold-averaged across seeds then 3-fold equal-weighted",
            "fold_point_estimates": fold_point_means,
            "checkpoint_first_3fold_equal_weight_ba": overall_ba,
            "primary_contrast_delta_21": {
                "point_estimate_pp": float(delta_21 * 100),
                "ci_95_pp": [float(boot["delta_21_ci_95"][0] * 100), float(boot["delta_21_ci_95"][1] * 100)],
                "significant_at_005": boot["delta_21_significant"],
            },
            "auxiliary_contrast_delta_10": {
                "point_estimate_pp": float(delta_10 * 100),
                "ci_95_pp": [float(boot["delta_10_ci_95"][0] * 100), float(boot["delta_10_ci_95"][1] * 100)],
                "significant_at_005": boot["delta_10_significant"],
            },
            "model_ba_ci_95": {m: [float(ci[0]*100), float(ci[1]*100)] for m, ci in boot["model_ba_ci_95"].items()},
            "raw_evaluations": raw_evals,
        }

        print(f"  {c_name} Results:")
        for m in MODELS:
            print(f"    {m:<10}: Overall BA = {overall_ba[m]*100:.2f}%, 95% CI: [{boot['model_ba_ci_95'][m][0]*100:.2f}, {boot['model_ba_ci_95'][m][1]*100:.2f}]%")
        print(f"    Delta_21* (FI - FC): {delta_21*100:+.2f} pp, 95% CI: [{boot['delta_21_ci_95'][0]*100:+.2f}, {boot['delta_21_ci_95'][1]*100:+.2f}] pp (Sig: {boot['delta_21_significant']})")
        print(f"    Delta_10* (FC - GP): {delta_10*100:+.2f} pp, 95% CI: [{boot['delta_10_ci_95'][0]*100:+.2f}, {boot['delta_10_ci_95'][1]*100:+.2f}] pp (Sig: {boot['delta_10_significant']})")

    results["evaluation_provenance"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL FROZEN EXTERNAL DIAGNOSTICS COMPLETED SUCCESSFULLY")
    print(f"Saved comprehensive diagnostic report to: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--threshold-file", type=Path, default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_external_diagnostics(args.models_dir, args.threshold_file, args.output, args.device)


if __name__ == "__main__":
    main()
