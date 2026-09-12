#!/usr/bin/env python3
"""Run P1 and P2 robustness analyses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.matched_support import compute_fc_support_14x14, compute_fi_support_14x14
from models.r2bda_net import R2BDANet


_dataset_spec = importlib.util.spec_from_file_location(
    "building_dataset", REPO_ROOT / "scripts" / "10_building_dataset.py"
)
_dataset_module = importlib.util.module_from_spec(_dataset_spec)
assert _dataset_spec.loader is not None
_dataset_spec.loader.exec_module(_dataset_module)
_normalise_image = _dataset_module._normalise_image
_resolve_mask_path = _dataset_module._resolve_mask_path
_to_hwc = _dataset_module._to_hwc


DATA_ROOT = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "results" / "models"
DEFAULT_THRESHOLD_FILE = (
    REPO_ROOT / "results" / "unified_global_threshold.json"
)
DEFAULT_ADDENDUM_FILE = REPO_ROOT / "configs" / "robustness_protocol.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "robustness_inference_v1"

CITIES = ("antakya", "nurdagi", "kahramanmaras")
SEEDS = (42, 43, 44)
ALL_MODELS = ("GP", "FC", "FI")
EXPERIMENT_MODELS = {
    "R1_footprint_error": ("FI",),
    "R2_candidate_localization_error": ("FC", "FI"),
}
FROZEN_TAU = 0.48
STRUCTURE_3X3 = np.ones((3, 3), dtype=bool)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metadata(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate_frozen_inputs(
    models_dir: Path, threshold_file: Path, addendum_file: Path
) -> tuple[dict, dict, float]:
    """Validate that runtime inputs exactly match the frozen addendum."""
    if not addendum_file.is_file():
        raise FileNotFoundError(f"Missing locked robustness addendum: {addendum_file}")
    addendum = read_json(addendum_file)
    if addendum.get("status") != "POST_HOC_PROTOCOL":
        raise ValueError("Robustness protocol must have status POST_HOC_PROTOCOL")

    threshold_info = read_json(threshold_file)
    tau = float(threshold_info["optimal_threshold_tau_star"])
    frozen_input = addendum["frozen_inputs"]
    if not math.isclose(tau, FROZEN_TAU, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Threshold file contains tau={tau}, expected frozen tau={FROZEN_TAU}")
    if not math.isclose(
        tau, float(frozen_input["threshold_tau_star"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Threshold file and robustness addendum disagree")

    selection_path = models_dir.parent / "selected_recipe_manifest.json"
    selection = read_json(selection_path)
    selected_recipe = selection.get("selected_recipe_id")
    if selected_recipe != frozen_input["selected_recipe_id"]:
        raise ValueError(
            f"Selected handoff recipe is {selected_recipe!r}; addendum freezes "
            f"{frozen_input['selected_recipe_id']!r}"
        )

    expected_models = {
        key: tuple(value["models"]) for key, value in addendum["experiments"].items()
    }
    if expected_models != EXPERIMENT_MODELS:
        raise ValueError(f"Unexpected addendum experiment/model matrix: {expected_models}")
    return addendum, selection, tau


def build_provenance(
    models_dir: Path,
    threshold_file: Path,
    addendum_file: Path,
    metadata_files: Iterable[Path],
    tau: float,
) -> dict:
    checkpoints = []
    for model_name in ALL_MODELS:
        for fold in CITIES:
            for seed in SEEDS:
                path = models_dir / model_name / fold / f"seed_{seed}" / "best.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing selected checkpoint: {path}")
                checkpoints.append(
                    {
                        "model": model_name,
                        "target_city_fold": fold,
                        "seed": seed,
                        "path": str(path),
                        "resolved_path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )

    selection_path = models_dir.parent / "selected_recipe_manifest.json"
    status = git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": git_value("rev-parse", "HEAD"),
            "status_porcelain": status,
            "dirty": status not in ("", "UNAVAILABLE"),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "addendum": {"path": str(addendum_file), "sha256": sha256_file(addendum_file)},
        "threshold": {
            "path": str(threshold_file),
            "sha256": sha256_file(threshold_file),
            "tau_star": tau,
        },
        "selected_recipe_manifest": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "metadata_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in metadata_files
        ],
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
    }


def shift_zero_filled(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a 2-D array without interpolation, wrapping, or edge reuse."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got {array.shape}")
    height, width = array.shape
    output = np.zeros_like(array)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return output


def translate_image_reflect(image: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate CHW image content, filling newly exposed pixels by reflection."""
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image.shape)}")
    if dy == 0 and dx == 0:
        return image.clone()
    height, width = image.shape[-2:]
    pad_y, pad_x = abs(dy), abs(dx)
    if pad_y >= height or pad_x >= width:
        raise ValueError("Reflection padding must be smaller than the image dimensions")
    padded = F.pad(image.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="reflect")[0]
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[:, y0 : y0 + height, x0 : x0 + width]


def deterministic_shift(cohort_id: str, building_id: str, distance: int) -> tuple[int, int]:
    """Return one cohort/building-specific direction shared by all magnitudes."""
    if distance < 0:
        raise ValueError("Shift distance must be non-negative")
    if distance == 0:
        return 0, 0
    key = f"matched_support_robustness_v1:{cohort_id}:{building_id}"
    fraction = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) / float(2**256)
    angle = fraction * 2.0 * math.pi
    dx = int(round(distance * math.cos(angle)))
    dy = int(round(distance * math.sin(angle)))
    return dy, dx


def one_pixel_centroid_fallback(original_mask: np.ndarray) -> np.ndarray:
    """Return one pixel at the half-up rounded centroid of a non-empty mask."""
    rows, cols = np.where(original_mask > 0)
    if len(rows) == 0:
        raise ValueError("Natural empty masks are not eligible for induced-empty fallback")
    row = int(np.floor(float(rows.mean()) + 0.5))
    col = int(np.floor(float(cols.mean()) + 0.5))
    row = int(np.clip(row, 0, original_mask.shape[0] - 1))
    col = int(np.clip(col, 0, original_mask.shape[1] - 1))
    fallback = np.zeros_like(original_mask, dtype=np.float32)
    fallback[row, col] = 1.0
    return fallback


def perturb_footprint_mask(
    original_mask: np.ndarray,
    state: dict,
    cohort_id: str,
    building_id: str,
) -> tuple[np.ndarray, dict]:
    """Apply one frozen R1 vector perturbation and its empty-erosion policy."""
    if int((original_mask > 0).sum()) == 0:
        raise ValueError(f"Natural empty footprint for building_id={building_id}")
    state_type = state["type"]
    dy = dx = 0
    if state_type == "clean":
        perturbed = original_mask.copy()
    elif state_type == "zero_filled_deterministic_shift":
        dy, dx = deterministic_shift(cohort_id, building_id, int(state["pixels"]))
        perturbed = shift_zero_filled(original_mask, dy, dx)
    elif state_type == "binary_erosion_3x3":
        perturbed = ndimage.binary_erosion(
            original_mask > 0, structure=STRUCTURE_3X3, iterations=int(state["iterations"])
        ).astype(np.float32)
    elif state_type == "binary_dilation_3x3":
        perturbed = ndimage.binary_dilation(
            original_mask > 0, structure=STRUCTURE_3X3, iterations=int(state["iterations"])
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported R1 perturbation type: {state_type}")

    induced_empty = bool((perturbed > 0).sum() == 0)
    fallback_applied = induced_empty and state_type == "binary_erosion_3x3"
    if fallback_applied:
        perturbed = one_pixel_centroid_fallback(original_mask)
    elif induced_empty:
        raise ValueError(
            f"Perturbation {state['id']} emptied building_id={building_id}; "
            "the locked fallback applies only to erosion"
        )
    return perturbed.astype(np.float32), {
        "requested_shift_dy": dy,
        "requested_shift_dx": dx,
        "induced_empty_mask": induced_empty,
        "centroid_fallback_applied": fallback_applied,
        "perturbed_mask_pixels": int((perturbed > 0).sum()),
    }


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_bin, second_bin = first > 0, second > 0
    union = int(np.logical_or(first_bin, second_bin).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(first_bin, second_bin).sum() / union)


class BaseRobustnessDataset(Dataset):
    """Load clean normalized images and raw footprint masks at model resolution."""

    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict("records")
        self.transform = A.Compose([A.Resize(224, 224), ToTensorV2()])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = DATA_ROOT / str(record["image_path"])
        if image_path.suffix.lower() in (".tif", ".tiff"):
            image = _to_hwc(tifffile.imread(image_path))
        else:
            from PIL import Image

            image = np.asarray(Image.open(image_path).convert("RGB"))
        if image.shape[-1] != 3:
            raise ValueError(f"Expected three image channels at {image_path}, got {image.shape}")
        image = _normalise_image(image, "global_percentile_1_99_v1")

        mask_path = Path(_resolve_mask_path(DATA_ROOT, record))
        mask = np.squeeze(tifffile.imread(mask_path))
        if mask.ndim != 2:
            raise ValueError(f"Expected 2-D footprint mask at {mask_path}, got {mask.shape}")
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask shape mismatch for building_id={record['building_id']}")
        mask = (mask > 0).astype(np.float32)
        if int(mask.sum()) == 0:
            raise ValueError(f"Natural empty footprint for building_id={record['building_id']}")

        transformed = self.transform(image=image, mask=mask)
        resized_mask = transformed["mask"]
        if torch.is_tensor(resized_mask):
            resized_mask = resized_mask.cpu().numpy()
        resized_mask = (np.squeeze(resized_mask) > 0.5).astype(np.float32)
        if int(resized_mask.sum()) == 0:
            raise ValueError(f"Footprint vanished during resize for building_id={record['building_id']}")

        label = record.get("label_2class", -1)
        if pd.isna(label) or int(label) not in (0, 1):
            raise ValueError(f"Invalid binary label for building_id={record['building_id']}: {label}")
        return {
            "image": transformed["image"].float(),
            "mask": torch.from_numpy(resized_mask),
            "label": int(label),
            "building_id": str(record["building_id"]),
        }


def preserve_requested_order(frame: pd.DataFrame, building_ids: list[str]) -> pd.DataFrame:
    """Select one row per requested building and preserve manifest order."""
    work = frame.copy()
    work["_bid"] = work["building_id"].astype(str)
    if work["_bid"].duplicated().any():
        duplicated = work.loc[work["_bid"].duplicated(), "_bid"].iloc[0]
        raise ValueError(f"Duplicate metadata row for building_id={duplicated}")
    indexed = work.set_index("_bid", drop=False)
    missing = [bid for bid in building_ids if bid not in indexed.index]
    if missing:
        raise KeyError(f"Metadata is missing {len(missing)} requested buildings; first={missing[0]}")
    selected = indexed.loc[building_ids].reset_index(drop=True)
    return selected.drop(columns=["_bid"])


def load_cohort_frames(addendum: dict) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[Path]]:
    """Build the exact Turkey city cohorts and Venezuela historical cohort."""
    turkey_path = DATA_ROOT / "09_metadata" / "turkey_worldview_loco_v3_base.parquet"
    venezuela_path = DATA_ROOT / "09_metadata" / "model_ready_patch_index_venezuela_wv.csv"
    turkey = read_metadata(turkey_path)
    venezuela = read_metadata(venezuela_path).reset_index(drop=True)

    frames: dict[str, pd.DataFrame] = {}
    sensors: dict[str, str] = {}
    turkey_total = 0
    for city in CITIES:
        manifest_path = DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"
        manifest = read_json(manifest_path)
        bids = [str(value) for value in manifest["target_buildings"]["test"]]
        frame = preserve_requested_order(turkey, bids)
        cohort_id = f"Turkey_LOCO_primary/{city}"
        frames[cohort_id] = frame
        unique_sensors = frame["sensor"].dropna().unique().tolist()
        if unique_sensors != ["turkey_wv_visual_rgb"]:
            raise ValueError(f"Unexpected Turkey sensor set for {city}: {unique_sensors}")
        sensors[cohort_id] = unique_sensors[0]
        turkey_total += len(frame)

    expected_turkey = int(addendum["cohorts"]["Turkey_LOCO_primary"]["n_samples"])
    if turkey_total != expected_turkey:
        raise ValueError(f"Turkey cohort count {turkey_total} != frozen count {expected_turkey}")
    expected_venezuela = int(
        addendum["cohorts"]["Venezuela_historical_external"]["n_samples"]
    )
    if len(venezuela) != expected_venezuela:
        raise ValueError(f"Venezuela cohort count {len(venezuela)} != frozen count {expected_venezuela}")
    venezuela_id = "Venezuela_historical_external/all"
    frames[venezuela_id] = venezuela
    unique_venezuela_sensors = venezuela["sensor"].dropna().unique().tolist()
    if len(unique_venezuela_sensors) != 1:
        raise ValueError(f"Expected one Venezuela sensor, got {unique_venezuela_sensors}")
    sensors[venezuela_id] = unique_venezuela_sensors[0]

    manifest_paths = [
        DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"
        for city in CITIES
    ]
    return frames, sensors, [turkey_path, venezuela_path, *manifest_paths]


def load_model(checkpoint_path: Path, sensor_name: str, device: torch.device) -> R2BDANet:
    net = R2BDANet(
        {sensor_name: 3},
        adapter_dim=3,
        backbone_name="swin_tiny",
        pretrained_backbone=False,
        mask_mode="matched_support_14x14",
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model"])
    net.eval()
    return net


def routing_for_cohort(cohort_id: str) -> tuple[str, ...]:
    if cohort_id.startswith("Turkey_LOCO_primary/"):
        return (cohort_id.rsplit("/", 1)[1],)
    return CITIES


def state_inputs(
    experiment: str,
    state: dict,
    model_name: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    cohort_id: str,
    building_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Build perturbed images/supports and per-building audit metadata."""
    out_images: list[torch.Tensor] = []
    supports: list[torch.Tensor] = []
    audits: list[dict] = []
    for image_tensor, mask_tensor, building_id in zip(images, masks, building_ids):
        original = mask_tensor.cpu().numpy().astype(np.float32)
        if experiment == "R1_footprint_error":
            perturbed, audit = perturb_footprint_mask(original, state, cohort_id, building_id)
            out_image = image_tensor
            support = compute_fi_support_14x14(perturbed)
        elif experiment == "R2_candidate_localization_error":
            distance = int(state["pixels"])
            dy, dx = deterministic_shift(cohort_id, building_id, distance)
            out_image = translate_image_reflect(image_tensor, dy, dx)
            translated_mask = shift_zero_filled(original, dy, dx)
            if int((translated_mask > 0).sum()) == 0:
                raise ValueError(
                    f"Localization state {state['id']} emptied building_id={building_id}"
                )
            support = (
                compute_fc_support_14x14()
                if model_name == "FC"
                else compute_fi_support_14x14(translated_mask)
            )
            perturbed = translated_mask
            audit = {
                "requested_shift_dy": dy,
                "requested_shift_dx": dx,
                "induced_empty_mask": False,
                "centroid_fallback_applied": False,
                "perturbed_mask_pixels": int((translated_mask > 0).sum()),
            }
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
        out_images.append(out_image)
        supports.append(torch.from_numpy(support).float())
        audits.append(
            {
                **audit,
                "original_mask_pixels": int((original > 0).sum()),
                "mask_iou_to_clean": mask_iou(original, perturbed),
            }
        )
    return torch.stack(out_images), torch.stack(supports), audits


class PredictionStore:
    """In-memory arrays required for checkpoint-first aggregation and bootstrap."""

    def __init__(self) -> None:
        self.y_true: dict[str, np.ndarray] = {}
        self.building_ids: dict[str, list[str]] = {}
        self.chunks: dict[tuple[str, str, str, str, str, int], list[np.ndarray]] = {}

    def append(
        self,
        cohort_id: str,
        experiment: str,
        state_id: str,
        model: str,
        fold: str,
        seed: int,
        values: np.ndarray,
    ) -> None:
        key = (experiment, state_id, model, fold, seed)
        self.chunks.setdefault((cohort_id, *key), []).append(values.astype(np.uint8))

    def finalize(self) -> dict[tuple[str, str, str, str, str, int], np.ndarray]:
        return {key: np.concatenate(parts) for key, parts in self.chunks.items()}


def _write_prediction_batch(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def preflight_cohort(
    cohort_id: str,
    frame: pd.DataFrame,
    states_by_experiment: dict[str, list[dict]],
    batch_size: int,
    num_workers: int,
) -> dict:
    """Validate all registered masks/supports on CPU before any model is loaded."""
    loader = DataLoader(
        BaseRobustnessDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    r1_empty_counts = {
        state["id"]: 0 for state in states_by_experiment["R1_footprint_error"]
    }
    r2_empty_counts = {
        state["id"]: 0
        for state in states_by_experiment["R2_candidate_localization_error"]
    }
    r2_empty_examples: list[str] = []
    n_seen = 0
    for batch in loader:
        masks = batch["mask"]
        building_ids = list(batch["building_id"])
        for mask_tensor, building_id in zip(masks, building_ids):
            original = mask_tensor.numpy().astype(np.float32)
            for state in states_by_experiment["R1_footprint_error"]:
                perturbed, audit = perturb_footprint_mask(
                    original, state, cohort_id, building_id
                )
                compute_fi_support_14x14(perturbed)
                r1_empty_counts[state["id"]] += int(audit["induced_empty_mask"])
            for state in states_by_experiment["R2_candidate_localization_error"]:
                distance = int(state["pixels"])
                dy, dx = deterministic_shift(cohort_id, building_id, distance)
                translated = shift_zero_filled(original, dy, dx)
                if int((translated > 0).sum()) == 0:
                    r2_empty_counts[state["id"]] += 1
                    if len(r2_empty_examples) < 10:
                        r2_empty_examples.append(f"{state['id']}:{building_id}")
                else:
                    compute_fi_support_14x14(translated)
            n_seen += 1

    r2_total_empty = sum(r2_empty_counts.values())
    if r2_total_empty:
        raise ValueError(
            f"CPU preflight found {r2_total_empty} unregistered empty R2 co-translated "
            f"masks in {cohort_id}; counts={r2_empty_counts}; "
            f"examples={r2_empty_examples}"
        )
    if n_seen != len(frame):
        raise AssertionError(f"CPU preflight saw {n_seen}/{len(frame)} rows for {cohort_id}")
    return {
        "n_samples_checked": n_seen,
        "R1_induced_empty_erosion_counts": r1_empty_counts,
        "R2_co_translated_empty_counts": r2_empty_counts,
        "status": "PASSED",
    }


def run_cohort_inference(
    cohort_id: str,
    frame: pd.DataFrame,
    sensor_name: str,
    states_by_experiment: dict[str, list[dict]],
    models_dir: Path,
    tau: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prediction_temp_path: Path,
    store: PredictionStore,
    parquet_writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    """Run one checkpoint at a time to bound GPU memory on 8 GB devices."""
    dataset = BaseRobustnessDataset(frame)
    folds = routing_for_cohort(cohort_id)
    checkpoint_number = 0
    checkpoint_total = 2 * len(folds) * len(SEEDS)
    for model_name in ("FC", "FI"):
        relevant_experiments = [
            name for name, names in EXPERIMENT_MODELS.items() if model_name in names
        ]
        for fold in folds:
            for seed in SEEDS:
                checkpoint_number += 1
                checkpoint_path = models_dir / model_name / fold / f"seed_{seed}" / "best.pt"
                model = load_model(checkpoint_path, sensor_name, device)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                )
                labels_parts: list[np.ndarray] = []
                bid_parts: list[str] = []
                for batch_index, batch in enumerate(loader):
                    clean_images = batch["image"]
                    clean_masks = batch["mask"]
                    labels = batch["label"].numpy().astype(np.int8)
                    building_ids = list(batch["building_id"])
                    labels_parts.append(labels)
                    bid_parts.extend(building_ids)
                    output_frames: list[pd.DataFrame] = []

                    for experiment in relevant_experiments:
                        for state in states_by_experiment[experiment]:
                            input_images, supports, audits = state_inputs(
                                experiment,
                                state,
                                model_name,
                                clean_images,
                                clean_masks,
                                cohort_id,
                                building_ids,
                            )
                            input_images = input_images.to(device, non_blocking=True)
                            supports = supports.to(device, non_blocking=True)
                            with torch.no_grad(), torch.amp.autocast(
                                device_type="cuda", enabled=device.type == "cuda"
                            ):
                                logits = model(input_images, sensor_name, mask=supports)[
                                    "logits_2class"
                                ]
                                probabilities = torch.softmax(logits, dim=1)[:, 1]
                            probs = probabilities.float().cpu().numpy()
                            predictions = (probs >= tau).astype(np.uint8)
                            store.append(
                                cohort_id,
                                experiment,
                                state["id"],
                                model_name,
                                fold,
                                seed,
                                predictions,
                            )
                            output_frames.append(
                                pd.DataFrame(
                                    {
                                        "cohort": cohort_id,
                                        "experiment": experiment,
                                        "state": state["id"],
                                        "model": model_name,
                                        "target_city_fold": fold,
                                        "seed": seed,
                                        "building_id": building_ids,
                                        "label_2class": labels,
                                        "probability_damaged": probs.astype(np.float32),
                                        "prediction_2class": predictions,
                                        "threshold_tau": np.float32(tau),
                                        "requested_shift_dy": [
                                            audit["requested_shift_dy"] for audit in audits
                                        ],
                                        "requested_shift_dx": [
                                            audit["requested_shift_dx"] for audit in audits
                                        ],
                                        "induced_empty_mask": [
                                            audit["induced_empty_mask"] for audit in audits
                                        ],
                                        "centroid_fallback_applied": [
                                            audit["centroid_fallback_applied"] for audit in audits
                                        ],
                                        "original_mask_pixels": [
                                            audit["original_mask_pixels"] for audit in audits
                                        ],
                                        "perturbed_mask_pixels": [
                                            audit["perturbed_mask_pixels"] for audit in audits
                                        ],
                                        "mask_iou_to_clean": np.asarray(
                                            [audit["mask_iou_to_clean"] for audit in audits],
                                            dtype=np.float32,
                                        ),
                                    }
                                )
                            )
                            del input_images, supports
                    parquet_writer = _write_prediction_batch(
                        parquet_writer,
                        pd.concat(output_frames, ignore_index=True),
                        prediction_temp_path,
                    )
                pass_labels = np.concatenate(labels_parts)
                if cohort_id not in store.y_true:
                    store.y_true[cohort_id] = pass_labels
                    store.building_ids[cohort_id] = bid_parts
                else:
                    np.testing.assert_array_equal(store.y_true[cohort_id], pass_labels)
                    if store.building_ids[cohort_id] != bid_parts:
                        raise AssertionError(f"Building order changed in {cohort_id}")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"  {cohort_id}: checkpoint {checkpoint_number}/{checkpoint_total} "
                    f"({model_name}, {fold}, seed={seed})"
                )

    for key, parts in store.chunks.items():
        if key[0] == cohort_id and sum(len(part) for part in parts) != len(dataset):
            raise AssertionError(f"Incomplete prediction array for {key}")
    assert parquet_writer is not None
    return parquet_writer


def binary_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy specialized to a two-class, both-classes-present sample."""
    intact = y_true == 0
    damaged = y_true == 1
    if not intact.any() or not damaged.any():
        raise ValueError("Balanced accuracy requires both classes")
    specificity = float((y_pred[intact] == 0).mean())
    sensitivity = float((y_pred[damaged] == 1).mean())
    return 0.5 * (specificity + sensitivity)


def checkpoint_first_score(
    cohort_ids: list[str],
    predictions: dict[tuple[str, str, str, str, str, int], np.ndarray],
    y_true: dict[str, np.ndarray],
    experiment: str,
    state: str,
    model: str,
    indices: dict[str, np.ndarray] | None = None,
) -> tuple[float, dict]:
    """Seed-average inside fold/group, then equal-weight the frozen groups."""
    group_scores: dict[str, dict] = {}
    for cohort_id in cohort_ids:
        idx = slice(None) if indices is None else indices[cohort_id]
        truth = y_true[cohort_id][idx]
        fold_scores = []
        fold_details = {}
        for fold in routing_for_cohort(cohort_id):
            seed_scores = []
            for seed in SEEDS:
                pred = predictions[(cohort_id, experiment, state, model, fold, seed)][idx]
                seed_scores.append(binary_balanced_accuracy(truth, pred))
            fold_score = float(np.mean(seed_scores))
            fold_scores.append(fold_score)
            fold_details[fold] = {
                "balanced_accuracy": fold_score,
                "seed_balanced_accuracies": seed_scores,
            }
        group_scores[cohort_id] = {
            "balanced_accuracy": float(np.mean(fold_scores)),
            "folds": fold_details,
        }
    return float(np.mean([item["balanced_accuracy"] for item in group_scores.values()])), group_scores


def stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    intact = np.flatnonzero(y_true == 0)
    damaged = np.flatnonzero(y_true == 1)
    if len(intact) == 0 or len(damaged) == 0:
        raise ValueError("Each bootstrap stratum must contain both binary classes")
    return np.concatenate(
        (
            rng.choice(intact, size=len(intact), replace=True),
            rng.choice(damaged, size=len(damaged), replace=True),
        )
    )


def summarize_domain(
    domain_name: str,
    cohort_ids: list[str],
    predictions: dict[tuple[str, str, str, str, str, int], np.ndarray],
    y_true: dict[str, np.ndarray],
    states_by_experiment: dict[str, list[dict]],
    n_boot: int,
    bootstrap_seed: int,
    practical_margin_pp: float,
) -> dict:
    """Compute checkpoint-first points and common-draw paired bootstrap intervals."""
    score_keys = []
    for experiment, states in states_by_experiment.items():
        for state in states:
            for model in EXPERIMENT_MODELS[experiment]:
                score_keys.append((experiment, state["id"], model))

    point_scores = {}
    point_details = {}
    for key in score_keys:
        point_scores[key], point_details[key] = checkpoint_first_score(
            cohort_ids, predictions, y_true, *key
        )

    rng = np.random.default_rng(bootstrap_seed)
    boot_scores = {key: np.empty(n_boot, dtype=np.float64) for key in score_keys}
    for iteration in range(n_boot):
        indices = {cohort: stratified_indices(y_true[cohort], rng) for cohort in cohort_ids}
        for key in score_keys:
            boot_scores[key][iteration] = checkpoint_first_score(
                cohort_ids, predictions, y_true, *key, indices=indices
            )[0]

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5) * 100), float(np.percentile(values, 97.5) * 100)]

    r1_states = states_by_experiment["R1_footprint_error"]
    clean_r1 = ("R1_footprint_error", "clean", "FI")
    r1_summary = {}
    for state in r1_states:
        key = ("R1_footprint_error", state["id"], "FI")
        delta = point_scores[key] - point_scores[clean_r1]
        boot_delta = boot_scores[key] - boot_scores[clean_r1]
        r1_summary[state["id"]] = {
            "state_definition": state,
            "FI_balanced_accuracy_pct": point_scores[key] * 100,
            "FI_ba_ci_95_pct": interval(boot_scores[key]),
            "delta_from_clean_pp": delta * 100,
            "delta_from_clean_ci_95_pp": interval(boot_delta),
            "checkpoint_first_details": point_details[key],
        }

    critical_shift_state = None
    for state in r1_states:
        if state["type"] != "zero_filled_deterministic_shift":
            continue
        record = r1_summary[state["id"]]
        if (
            record["delta_from_clean_pp"] <= -practical_margin_pp
            and record["delta_from_clean_ci_95_pp"][1] < -practical_margin_pp
        ):
            critical_shift_state = state["id"]
            break

    r2_states = states_by_experiment["R2_candidate_localization_error"]
    r2_summary = {}
    for state in r2_states:
        state_id = state["id"]
        model_records = {}
        for model in ("FC", "FI"):
            key = ("R2_candidate_localization_error", state_id, model)
            clean_key = ("R2_candidate_localization_error", "center_0px", model)
            delta = point_scores[key] - point_scores[clean_key]
            boot_delta = boot_scores[key] - boot_scores[clean_key]
            model_records[model] = {
                "balanced_accuracy_pct": point_scores[key] * 100,
                "ba_ci_95_pct": interval(boot_scores[key]),
                "delta_from_model_clean_pp": delta * 100,
                "delta_from_model_clean_ci_95_pp": interval(boot_delta),
                "checkpoint_first_details": point_details[key],
            }
        m1_key = ("R2_candidate_localization_error", state_id, "FC")
        m2_key = ("R2_candidate_localization_error", state_id, "FI")
        contrast_boot = boot_scores[m1_key] - boot_scores[m2_key]
        contrast_point = point_scores[m1_key] - point_scores[m2_key]
        r2_summary[state_id] = {
            "state_definition": state,
            "models": model_records,
            "FC_minus_FI_pp": contrast_point * 100,
            "FC_minus_FI_ci_95_pp": interval(contrast_boot),
        }

    clean_contrast = r2_summary["center_0px"]["FC_minus_FI_pp"]
    clean_lower = r2_summary["center_0px"]["FC_minus_FI_ci_95_pp"][0]
    r2_critical_shifts = {}
    for model in ("FC", "FI"):
        r2_critical_shifts[model] = None
        for state in r2_states:
            if int(state["pixels"]) == 0:
                continue
            record = r2_summary[state["id"]]["models"][model]
            if (
                record["delta_from_model_clean_pp"] <= -practical_margin_pp
                and record["delta_from_model_clean_ci_95_pp"][1] < -practical_margin_pp
            ):
                r2_critical_shifts[model] = state["id"]
                break
    return {
        "domain": domain_name,
        "n_samples": int(sum(len(y_true[cohort]) for cohort in cohort_ids)),
        "cohort_partitions": {cohort: int(len(y_true[cohort])) for cohort in cohort_ids},
        "aggregation": (
            "checkpoint-first; seed mean within target city then equal-weight three-city mean"
            if domain_name == "Turkey_LOCO_primary"
            else "checkpoint-first; seed mean within fold then equal-weight three-fold mean"
        ),
        "R1_footprint_error": {
            "states": r1_summary,
            "critical_shift_boundary_rule": (
                "smallest listed shift state with point delta <= -margin and paired "
                "95% CI upper bound < -margin"
            ),
            "practical_margin_pp": practical_margin_pp,
            "observed_critical_shift_state": critical_shift_state,
            "morphology_interpretation": "reported independently; not ordered into a critical boundary",
        },
        "R2_candidate_localization_error": {
            "states": r2_summary,
            "critical_shift_boundary_rule": (
                "per model, smallest non-zero localization shift with point delta <= -margin "
                "and paired 95% CI upper bound < -margin"
            ),
            "observed_critical_shift_state_by_model": r2_critical_shifts,
            "clean_practical_noninferiority_readout": {
                "contrast": "FC - FI",
                "point_estimate_pp": clean_contrast,
                "one_sided_97_5pct_lower_bound_pp": clean_lower,
                "margin_pp": -practical_margin_pp,
                "passes_exploratory_margin": bool(clean_lower > -practical_margin_pp),
            },
        },
        "bootstrap": {
            "iterations": n_boot,
            "seed": bootstrap_seed,
            "stratification": "within cohort partition and binary label",
            "pairing": "common resampled building indices across all models, checkpoints, and states",
        },
    }


def publish_without_overwrite(temp_path: Path, final_path: Path) -> None:
    """Atomically create a final hard link; fail if the destination already exists."""
    try:
        os.link(temp_path, final_path)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing output: {final_path}") from exc
    temp_path.unlink()


def run_robustness_study(
    models_dir: Path,
    threshold_file: Path,
    addendum_file: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
    n_boot: int,
    bootstrap_seed: int,
) -> None:
    addendum, selection, tau = validate_frozen_inputs(
        models_dir, threshold_file, addendum_file
    )
    if n_boot != 2000 or bootstrap_seed != 42:
        raise ValueError("The frozen protocol requires n_boot=2000 and bootstrap_seed=42")

    summary_path = output_dir / "robustness_summary.json"
    predictions_path = output_dir / "robustness_predictions.parquet"
    for path in (summary_path, predictions_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_suffix = f".tmp.{os.getpid()}"
    summary_temp = output_dir / f"robustness_summary.json{temp_suffix}"
    predictions_temp = output_dir / f"robustness_predictions.parquet{temp_suffix}"
    if summary_temp.exists() or predictions_temp.exists():
        raise FileExistsError("Temporary output collision; stop and inspect the output directory")

    frames, sensors, metadata_files = load_cohort_frames(addendum)
    provenance = build_provenance(
        models_dir, threshold_file, addendum_file, metadata_files, tau
    )
    states_by_experiment = {
        name: list(addendum["experiments"][name]["states"])
        for name in EXPERIMENT_MODELS
    }
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu"
    )
    if device_name == "cuda" and device.type != "cuda":
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    print("MATCHED-SUPPORT 14x14 ROBUSTNESS INFERENCE")
    print(f"  threshold tau*: {tau}")
    print(f"  selected recipe: {selection['selected_recipe_id']}")
    print(f"  device: {device}")
    print(f"  output directory: {output_dir}")

    preflight = {}
    for cohort_id, frame in frames.items():
        print(f"CPU preflight {cohort_id}: n={len(frame)}")
        preflight[cohort_id] = preflight_cohort(
            cohort_id, frame, states_by_experiment, batch_size, num_workers
        )

    store = PredictionStore()
    writer: pq.ParquetWriter | None = None
    try:
        for cohort_id, frame in frames.items():
            print(f"Evaluating {cohort_id}: n={len(frame)}")
            writer = run_cohort_inference(
                cohort_id,
                frame,
                sensors[cohort_id],
                states_by_experiment,
                models_dir,
                tau,
                device,
                batch_size,
                num_workers,
                predictions_temp,
                store,
                writer,
            )
        if writer is not None:
            writer.close()
            writer = None

        predictions = store.finalize()
        turkey_cohorts = [f"Turkey_LOCO_primary/{city}" for city in CITIES]
        venezuela_cohorts = ["Venezuela_historical_external/all"]
        margin = float(addendum["analysis"]["practical_margin_pp"])
        domains = {
            "Turkey_LOCO_primary": summarize_domain(
                "Turkey_LOCO_primary",
                turkey_cohorts,
                predictions,
                store.y_true,
                states_by_experiment,
                n_boot,
                bootstrap_seed,
                margin,
            ),
            "Venezuela_historical_external": summarize_domain(
                "Venezuela_historical_external",
                venezuela_cohorts,
                predictions,
                store.y_true,
                states_by_experiment,
                n_boot,
                bootstrap_seed,
                margin,
            ),
        }
        provenance["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result = {
            "schema_version": 1,
            "protocol": addendum["protocol"],
            "status": "COMPLETED",
            "unified_global_threshold_tau_star": tau,
            "selected_recipe_id": selection["selected_recipe_id"],
            "provenance": provenance,
            "raw_predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_temp),
                "format": "Parquet; one row per building x state x checkpoint",
            },
            "cpu_preflight": preflight,
            "domains": domains,
        }
        with summary_temp.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

        publish_without_overwrite(predictions_temp, predictions_path)
        publish_without_overwrite(summary_temp, summary_path)
    except BaseException:
        if writer is not None:
            writer.close()
        raise

    print(f"Saved raw predictions: {predictions_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--threshold-file", type=Path, default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--addendum-file", type=Path, default=DEFAULT_ADDENDUM_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers must be non-negative")
    run_robustness_study(
        args.models_dir,
        args.threshold_file,
        args.addendum_file,
        args.output_dir,
        args.device,
        args.batch_size,
        args.num_workers,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
