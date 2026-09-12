#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "candidate_centering_audit.json"
)
FINAL_INPUT_SIZE = 224


def read_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def mask_centroid_offset(mask_path: Path) -> float:
    with rasterio.open(mask_path) as source:
        mask = source.read(1) > 0
    rows, columns = np.nonzero(mask)
    if columns.size == 0:
        raise ValueError(f"Empty footprint mask: {mask_path}")
    center_x = mask.shape[1] / 2.0
    center_y = mask.shape[0] / 2.0
    centroid_x = float(np.mean(columns + 0.5))
    centroid_y = float(np.mean(rows + 0.5))
    native_distance = float(np.hypot(centroid_x - center_x, centroid_y - center_y))
    if mask.shape[0] != mask.shape[1]:
        raise ValueError(f"Expected a square model-ready mask: {mask_path}")
    return native_distance * FINAL_INPUT_SIZE / mask.shape[1]


def describe(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "n": int(values.size),
        "mean_px": float(np.mean(values)),
        "p25_px": float(quantiles[0]),
        "median_px": float(quantiles[1]),
        "p75_px": float(quantiles[2]),
        "p90_px": float(quantiles[3]),
        "p95_px": float(quantiles[4]),
        "maximum_px": float(np.max(values)),
    }


def audit_index(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = read_index(path).copy()
    frame["mask_centroid_offset_224_px"] = [
        mask_centroid_offset(DATA_ROOT / mask_path) for mask_path in frame["mask_path"]
    ]
    report: dict[str, object] = {
        "overall": describe(frame["mask_centroid_offset_224_px"].to_numpy())
    }
    if "city" in frame.columns:
        report["by_city"] = {
            str(city): describe(group["mask_centroid_offset_224_px"].to_numpy())
            for city, group in frame.groupby("city", sort=True)
        }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    turkey_path = DATA_ROOT / "09_metadata" / "turkey_worldview_loco_v3_base.parquet"
    venezuela_path = DATA_ROOT / "09_metadata" / "model_ready_patch_index_venezuela_wv.csv"
    _, turkey = audit_index(turkey_path)
    _, venezuela = audit_index(venezuela_path)
    payload = {
        "role": "descriptive crop-geometry audit; not operational localization error",
        "distance_definition": (
            "Euclidean distance from the binary-mask pixel centroid to the patch "
            "center, scaled from the 256 x 256 model-ready mask to 224 x 224"
        ),
        "final_model_input_size": [FINAL_INPUT_SIZE, FINAL_INPUT_SIZE],
        "cohorts": {"Turkey": turkey, "Venezuela": venezuela},
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
