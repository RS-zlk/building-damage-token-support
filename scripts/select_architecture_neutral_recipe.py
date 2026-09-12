#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RECIPES_DIR = REPO_ROOT / "results" / "recipes"
DEFAULT_MODELS_DIR = REPO_ROOT / "results" / "models"
DEFAULT_SUMMARY_FILE = REPO_ROOT / "results" / "architecture_neutral_recipe_selection.json"
DEFAULT_MANIFEST_FILE = REPO_ROOT / "results" / "selected_recipe_manifest.json"

RECIPES = [
    "R1_CE_baseline",
    "R2_Focal_BS16",
    "R3_C4b_repaired_BS32",
]

MODELS = ["GP", "FC", "FI"]
CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]


def read_checkpoint_val_ba(ckpt_dir: Path) -> float:
    """Extract the best validation Balanced Accuracy from run artifacts."""
    meta_path = ckpt_dir / "run_metadata.json"
    history_path = ckpt_dir / "history.csv"
    best_pt = ckpt_dir / "best.pt"

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "best_validation_score" in meta:
                return float(meta["best_validation_score"])
        except Exception:
            pass

    if history_path.exists():
        try:
            df = pd.read_csv(history_path)
            if "binary_balanced_accuracy" in df.columns:
                return float(df["binary_balanced_accuracy"].max())
        except Exception:
            pass

    if best_pt.exists():
        try:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            val_metrics = ckpt.get("val_metrics", {})
            if "binary_balanced_accuracy" in val_metrics:
                return float(val_metrics["binary_balanced_accuracy"])
        except Exception:
            pass

    raise FileNotFoundError(f"Could not read validation score from {ckpt_dir}")


def evaluate_recipe(recipes_dir: Path, recipe_id: str) -> dict[str, object]:
    """Audit all 27 runs for a given recipe and compute summary metrics."""
    scores = {}
    missing_runs = []

    for m in MODELS:
        scores[m] = {}
        for c in CITIES:
            scores[m][c] = {}
            for s in SEEDS:
                run_dir = recipes_dir / recipe_id / m / c / f"seed_{s}"
                if not (run_dir / "best.pt").exists():
                    missing_runs.append(f"{recipe_id}/{m}/{c}/seed_{s}")
                    continue
                score = read_checkpoint_val_ba(run_dir)
                scores[m][c][s] = score

    if missing_runs:
        raise FileNotFoundError(
            f"Recipe '{recipe_id}' has {len(missing_runs)} missing runs out of 27!\n"
            f"First missing: {missing_runs[0]}"
        )

    all_27_scores = []
    model_means = {}

    for m in MODELS:
        m_scores = []
        for c in CITIES:
            for s in SEEDS:
                val = scores[m][c][s]
                all_27_scores.append(val)
                m_scores.append(val)
        model_means[m] = float(np.mean(m_scores))

    overall_score = float(np.mean(all_27_scores))
    cross_model_variance = float(np.var(list(model_means.values())))
    min_model_perf = float(np.min(list(model_means.values())))

    return {
        "recipe_id": recipe_id,
        "overall_score_mean_ba": overall_score,
        "model_means": model_means,
        "cross_model_variance": cross_model_variance,
        "min_model_performance": min_model_perf,
        "checkpoint_scores": scores,
    }


def select_winning_recipe(recipe_results: list[dict[str, object]]) -> tuple[dict[str, object], list[str]]:
    """Apply strict tie-breaking rules to select the winning recipe."""
    log_reasons = []

    sorted_by_score = sorted(
        recipe_results, key=lambda r: r["overall_score_mean_ba"], reverse=True
    )
    best_score = sorted_by_score[0]["overall_score_mean_ba"]

    candidates = [
        r for r in sorted_by_score if abs(r["overall_score_mean_ba"] - best_score) < 1e-6
    ]

    if len(candidates) == 1:
        winner = candidates[0]
        log_reasons.append(
            f"Rule 1 (Highest Score): '{winner['recipe_id']}' won with Score = {winner['overall_score_mean_ba']:.5f}."
        )
        return winner, log_reasons

    log_reasons.append(
        f"Score tie detected among {[c['recipe_id'] for c in candidates]} (Score ~ {best_score:.5f})."
    )

    sorted_by_var = sorted(candidates, key=lambda r: r["cross_model_variance"])
    best_var = sorted_by_var[0]["cross_model_variance"]
    candidates = [
        r for r in sorted_by_var if abs(r["cross_model_variance"] - best_var) < 1e-8
    ]

    if len(candidates) == 1:
        winner = candidates[0]
        log_reasons.append(
            f"Rule 2 (Lowest Cross-Model Variance): '{winner['recipe_id']}' won with Var = {winner['cross_model_variance']:.8f}."
        )
        return winner, log_reasons

    log_reasons.append(
        f"Variance tie detected among {[c['recipe_id'] for c in candidates]} (Var ~ {best_var:.8f})."
    )

    sorted_by_min = sorted(
        candidates, key=lambda r: r["min_model_performance"], reverse=True
    )
    best_min = sorted_by_min[0]["min_model_performance"]
    candidates = [
        r for r in sorted_by_min if abs(r["min_model_performance"] - best_min) < 1e-6
    ]

    if len(candidates) == 1:
        winner = candidates[0]
        log_reasons.append(
            f"Rule 3 (Highest Min-Model Performance): '{winner['recipe_id']}' won with Min = {winner['min_model_performance']:.5f}."
        )
        return winner, log_reasons

    preference_order = ["R3_C4b_repaired_BS32", "R2_Focal_BS16", "R1_CE_baseline"]
    winner = min(candidates, key=lambda r: preference_order.index(r["recipe_id"]))
    log_reasons.append(
        f"Rule 4 (Deterministic Order): '{winner['recipe_id']}' selected as fallback."
    )
    return winner, log_reasons


def link_winning_checkpoints(
    recipes_dir: Path, winning_recipe_id: str, target_models_dir: Path
):
    """Safely populate results/models/ with the winning recipe's checkpoints."""
    target_models_dir.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        for c in CITIES:
            for s in SEEDS:
                src_dir = recipes_dir / winning_recipe_id / m / c / f"seed_{s}"
                dst_dir = target_models_dir / m / c / f"seed_{s}"
                dst_dir.mkdir(parents=True, exist_ok=True)

                for fname in ["best.pt", "history.csv", "run_metadata.json"]:
                    src_file = src_dir / fname
                    dst_file = dst_dir / fname

                    if dst_file.exists() or dst_file.is_symlink():
                        dst_file.unlink()

                    if src_file.exists():
                        try:
                            rel_src = os.path.relpath(src_file, dst_dir)
                            dst_file.symlink_to(rel_src)
                        except OSError:
                            shutil.copy2(src_file, dst_file)


def run_strategy_selection(
    recipes_dir: Path,
    models_dir: Path,
    summary_file: Path,
    manifest_file: Path,
    dry_run: bool = False,
):
    print("=" * 80)
    print("PHASE C: ARCHITECTURE-NEUTRAL RECIPE STRATEGY SELECTION")
    print(f"Recipes Directory: {recipes_dir}")
    print(f"Destination Models Directory: {models_dir}")
    print("=" * 80)

    recipe_summaries = []
    for r in RECIPES:
        print(f"\nAuditing Recipe: {r} (27 checkpoints)...")
        summary = evaluate_recipe(recipes_dir, r)
        recipe_summaries.append(summary)
        print(f"  Overall Score (Mean BA): {summary['overall_score_mean_ba']*100:.2f}%")
        for m in MODELS:
            print(f"    {m:<10}: {summary['model_means'][m]*100:.2f}%")
        print(f"  Cross-Model Variance:    {summary['cross_model_variance']:.8f}")
        print(f"  Min Model Performance:   {summary['min_model_performance']*100:.2f}%")

    winner, decision_logs = select_winning_recipe(recipe_summaries)
    winning_id = winner["recipe_id"]

    print("\n" + "=" * 80)
    print(f"STRATEGY SELECTION WINNER: {winning_id}")
    print("Decision Rationale:")
    for log in decision_logs:
        print(f"  - {log}")
    print("=" * 80)

    selection_report = {
        "protocol": "Architecture-Neutral Strategy Selection (81 Runs)",
        "candidate_recipes": RECIPES,
        "selected_winning_recipe": winning_id,
        "decision_rationale": decision_logs,
        "ranking": [
            {
                "rank": idx + 1,
                "recipe_id": r["recipe_id"],
                "overall_score_mean_ba": r["overall_score_mean_ba"],
                "cross_model_variance": r["cross_model_variance"],
                "min_model_performance": r["min_model_performance"],
                "model_means": r["model_means"],
            }
            for idx, r in enumerate(
                sorted(recipe_summaries, key=lambda x: x["overall_score_mean_ba"], reverse=True)
            )
        ],
        "recipe_summaries": recipe_summaries,
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(selection_report, f, indent=2)
    print(f"\nSaved strategy selection summary to: {summary_file}")

    if not dry_run:
        print(f"Linking winning recipe ({winning_id}) checkpoints to: {models_dir}...")
        link_winning_checkpoints(recipes_dir, winning_id, models_dir)

        manifest = {
            "protocol": "Selected Architecture-Neutral Recipe Manifest",
            "selected_recipe_id": winning_id,
            "status": "LOCKED_AND_HANDED_OFF",
            "selection_mode": "81_run_architecture_neutral_selection",
            "source_recipe_dir": str(recipes_dir / winning_id),
            "models_dir": str(models_dir),
            "winning_score_mean_ba": winner["overall_score_mean_ba"],
            "model_family_means": winner["model_means"],
        }
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved handoff manifest to: {manifest_file}")
        print("Downstream scripts (threshold, primary eval, external diag) will read from this winning recipe.")
    else:
        print("[DRY-RUN] Skipped linking checkpoints and writing manifest.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes-dir", type=Path, default=DEFAULT_RECIPES_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_strategy_selection(
        args.recipes_dir,
        args.models_dir,
        args.summary_file,
        args.manifest_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
