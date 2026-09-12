#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MODELS = REPO_ROOT / "results" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "spatial_block_bootstrap_sensitivity.json"
MODELS = ["GP", "FC", "FI"]
CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    if specification.loader is None:
        raise ImportError(path)
    specification.loader.exec_module(module)
    return module


primary = load_module(
    "matched_support_primary_spatial",
    REPO_ROOT / "scripts" / "run_matched_support_primary_evaluation.py",
)
def building_centers(coordinates_file: Path) -> dict[int, tuple[float, float]]:
    frame = pd.read_csv(coordinates_file)
    required = {"building_id", "easting_m", "northing_m"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {sorted(missing)}")
    return {
        int(row.building_id): (float(row.easting_m), float(row.northing_m))
        for row in frame.itertuples(index=False)
    }


def collect_predictions(models_dir: Path, threshold: float, device: torch.device):
    results = {model: {city: {} for city in CITIES} for model in MODELS}
    for model in MODELS:
        for city in CITIES:
            for seed in SEEDS:
                checkpoint = models_dir / model / city / f"seed_{seed}" / "best.pt"
                results[model][city][seed] = primary.evaluate_single_checkpoint(
                    model, city, seed, checkpoint, threshold, device
                )
    return results


def block_bootstrap(
    results,
    centers: dict[int, tuple[float, float]],
    block_size_m: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    city_data = {}
    for city in CITIES:
        reference = results["GP"][city][SEEDS[0]]
        building_ids = [int(value) for value in reference["building_ids"]]
        y_true = np.asarray(reference["y_true"])
        missing = sorted(set(building_ids) - set(centers))
        if missing:
            raise KeyError(f"Missing coordinates for {len(missing)} buildings in {city}")
        block_labels = [
            (
                int(np.floor(centers[building_id][0] / block_size_m)),
                int(np.floor(centers[building_id][1] / block_size_m)),
            )
            for building_id in building_ids
        ]
        unique_blocks = sorted(set(block_labels))
        block_indices = {
            block: np.asarray(
                [index for index, value in enumerate(block_labels) if value == block],
                dtype=int,
            )
            for block in unique_blocks
        }
        predictions = {model: {} for model in MODELS}
        for model in MODELS:
            for checkpoint_seed in SEEDS:
                record = results[model][city][checkpoint_seed]
                if [int(value) for value in record["building_ids"]] != building_ids:
                    raise ValueError(f"Building-order mismatch: {model}, {city}, {checkpoint_seed}")
                predictions[model][checkpoint_seed] = np.asarray(record["y_pred"])
        city_data[city] = {
            "y_true": y_true,
            "blocks": unique_blocks,
            "block_indices": block_indices,
            "predictions": predictions,
        }

    fi_fc = []
    fc_gp = []
    rejected = 0
    attempts = 0
    while len(fi_fc) < n_bootstrap:
        attempts += 1
        if attempts > n_bootstrap * 10:
            raise RuntimeError("Too many one-class spatial bootstrap replicates")
        city_scores = {model: [] for model in MODELS}
        valid = True
        for city in CITIES:
            data = city_data[city]
            sampled_blocks = rng.choice(
                len(data["blocks"]), size=len(data["blocks"]), replace=True
            )
            indices = np.concatenate(
                [data["block_indices"][data["blocks"][value]] for value in sampled_blocks]
            )
            truth = data["y_true"][indices]
            if np.unique(truth).size != 2:
                valid = False
                break
            for model in MODELS:
                seed_scores = [
                    balanced_accuracy_score(
                        truth, data["predictions"][model][checkpoint_seed][indices]
                    )
                    for checkpoint_seed in SEEDS
                ]
                city_scores[model].append(float(np.mean(seed_scores)))
        if not valid:
            rejected += 1
            continue
        overall = {model: float(np.mean(values)) for model, values in city_scores.items()}
        fi_fc.append((overall["FI"] - overall["FC"]) * 100)
        fc_gp.append((overall["FC"] - overall["GP"]) * 100)

    return {
        "block_size_m": block_size_m,
        "blocks_per_city": {
            city: len(data["blocks"]) for city, data in city_data.items()
        },
        "bootstrap_iterations": n_bootstrap,
        "rejected_one_class_draws": rejected,
        "fi_minus_fc_ci_95_pp": [
            float(np.percentile(fi_fc, 2.5)),
            float(np.percentile(fi_fc, 97.5)),
        ],
        "fc_minus_gp_ci_95_pp": [
            float(np.percentile(fc_gp, 2.5)),
            float(np.percentile(fc_gp, 97.5)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coordinates-file",
        type=Path,
        default=DATA_ROOT / "09_metadata" / "turkey_building_coordinates_utm37n.csv",
        help="CSV with building_id, easting_m, and northing_m in UTM zone 37N.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[500, 1000, 2000])
    arguments = parser.parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and arguments.device != "cpu" else "cpu"
    )
    threshold_path = REPO_ROOT / "results" / "unified_global_threshold.json"
    threshold_value = float(json.loads(threshold_path.read_text())["optimal_threshold_tau_star"])
    results = collect_predictions(arguments.models_dir, threshold_value, device)
    centers = building_centers(arguments.coordinates_file)
    analyses = [
        block_bootstrap(results, centers, size, arguments.iterations, seed=42)
        for size in arguments.block_sizes
    ]
    payload = {
        "role": "post-hoc spatial-block bootstrap sensitivity",
        "primary_building_bootstrap_unchanged": True,
        "coordinate_system": "UTM zone 37N",
        "grid_origin": "projected CRS origin",
        "common_threshold": threshold_value,
        "checkpoint_first_equal_city_estimand": True,
        "analyses": analyses,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
