#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPO_ROOT / "results" / "models"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "matched_support_v1"
    / "method_specific_threshold_sensitivity.json"
)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    if specification.loader is None:
        raise ImportError(path)
    specification.loader.exec_module(module)
    return module


threshold = load_module(
    "matched_support_threshold",
    REPO_ROOT / "scripts" / "select_unified_global_threshold.py",
)
primary = load_module(
    "matched_support_primary",
    REPO_ROOT / "scripts" / "run_matched_support_primary_evaluation.py",
)


def run(models_dir: Path, output: Path, device_name: str) -> dict[str, object]:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu"
    )
    models = list(threshold.MODELS)
    cities = list(threshold.CITIES)
    seeds = list(threshold.SEEDS)
    taus = np.arange(0.10, 0.91, 0.01)

    method_thresholds: dict[str, float] = {}
    source_scores: dict[str, object] = {}
    for model in models:
        checkpoint_data = []
        for city in cities:
            for seed in seeds:
                checkpoint = models_dir / model / city / f"seed_{seed}" / "best.pt"
                y_true, y_prob = threshold.load_val_predictions_for_checkpoint(
                    model, city, seed, checkpoint, device
                )
                checkpoint_data.append((y_true, y_prob))
        curve = []
        for tau in taus:
            values = [
                threshold.balanced_accuracy_score(
                    y_true, (y_prob >= tau).astype(int)
                )
                for y_true, y_prob in checkpoint_data
            ]
            curve.append(
                {
                    "tau": round(float(tau), 2),
                    "mean_source_val_ba": float(np.mean(values)),
                }
            )
        best, reasons = threshold.select_best_threshold(curve)
        method_thresholds[model] = float(best["tau"])
        source_scores[model] = {
            "mean_source_val_ba": float(best["mean_source_val_ba"]),
            "decision_rationale": reasons,
        }

    checkpoint_results = {model: {city: {} for city in cities} for model in models}
    for model in models:
        for city in cities:
            for seed in seeds:
                checkpoint = models_dir / model / city / f"seed_{seed}" / "best.pt"
                checkpoint_results[model][city][seed] = primary.evaluate_single_checkpoint(
                    model,
                    city,
                    seed,
                    checkpoint,
                    method_thresholds[model],
                    device,
                )

    per_city = {model: {} for model in models}
    overall = {}
    for model in models:
        for city in cities:
            per_city[model][city] = float(
                np.mean(
                    [
                        checkpoint_results[model][city][seed]["balanced_accuracy"]
                        for seed in seeds
                    ]
                )
            )
        overall[model] = float(np.mean(list(per_city[model].values())))

    bootstrap = primary.compute_checkpoint_first_bootstrap(
        checkpoint_results, n_boot=2000, seed=42
    )
    payload = {
        "role": "post-hoc method-specific source-validation threshold sensitivity",
        "primary_protocol_unchanged": True,
        "threshold_search_grid": {"min": 0.10, "max": 0.90, "step": 0.01},
        "method_thresholds": method_thresholds,
        "source_validation": source_scores,
        "turkey_loco": {
            "overall_ba_percent": {
                model: value * 100 for model, value in overall.items()
            },
            "per_city_ba_percent": {
                model: {city: value * 100 for city, value in values.items()}
                for model, values in per_city.items()
            },
            "fi_minus_fc_pp": (overall["FI"] - overall["FC"]) * 100,
            "fc_minus_gp_pp": (overall["FC"] - overall["GP"]) * 100,
            "fi_minus_fc_ci_95_pp": [
                value * 100 for value in bootstrap["delta_21_ci_95"]
            ],
            "fc_minus_gp_ci_95_pp": [
                value * 100 for value in bootstrap["delta_10_ci_95"]
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.models_dir, arguments.output, arguments.device), indent=2))


if __name__ == "__main__":
    main()
