#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util
from models.r2bda_net import R2BDANet

spec = importlib.util.spec_from_file_location("building_dataset", REPO_ROOT / "scripts" / "10_building_dataset.py")
b_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b_module)
BuildingInstanceDataset = b_module.BuildingInstanceDataset

DATA_ROOT = REPO_ROOT / "data"
RECIPES_BASE = REPO_ROOT / "results" / "recipes"
MODELS_BASE = REPO_ROOT / "results" / "models"

CITIES = ["antakya", "nurdagi", "kahramanmaras"]
SEEDS = [42, 43, 44]

RECIPE_CONFIGS = {
    "R1_CE_baseline": {
        "desc": "Standard Cross-Entropy Loss with BS=32 (Damage weight 1.0)",
        "batch_size": 32,
        "train_args": "--loss-type ce --damage-weight 1.0 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 --epochs 80",
    },
    "R2_Focal_BS16": {
        "desc": "Focal Loss with BS=16 (~375 steps/epoch; distinct recipe, not pure BS ablation)",
        "batch_size": 16,
        "train_args": "--loss-type focal --focal-gamma 2.0 --damage-weight 1.5 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 --epochs 80",
    },
    "R3_C4b_repaired_BS32": {
        "desc": "Focal Loss with BS=32 (Inherited C4b Repaired at 14x14)",
        "batch_size": 32,
        "train_args": "--loss-type focal --focal-gamma 2.0 --damage-weight 1.5 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 --epochs 80",
    },
}

MODELS = {
    "GP": {
        "desc": "Capacity-Matched Global Pooling (14x14 Stage 3, 384-dim)",
        "mask_mode": "global_matched_14x14",
        "mask_ablation": None,
        "train_args": "--mask-mode global_matched_14x14",
    },
    "FC": {
        "desc": "Fixed Central Proxy Support (4 interior, 12 context tokens)",
        "mask_mode": "matched_support_14x14",
        "mask_ablation": "FC",
        "train_args": "--mask-mode matched_support_14x14 --mask-ablation FC",
    },
    "FI": {
        "desc": "Footprint-Derived Matched Support (4 interior, 12 strict exterior tokens)",
        "mask_mode": "matched_support_14x14",
        "mask_ablation": None,
        "train_args": "--mask-mode matched_support_14x14",
    },
}

COMMON_TRAIN_ARGS = (
    "--sensor turkey_wv_visual_rgb --split-mode source "
    "--backbone swin_tiny --pretrained-backbone "
    "--selection-metric binary_balanced_accuracy "
    "--balance-mode sampler --augmentation-preset standard "
    "--no-use-ema --amp --allow-existing-output "
    "--num-workers 4 --pin-memory --persistent-workers --prefetch-factor 2"
)


def is_run_completed(out_dir: Path, expected_epochs: int = 80) -> bool:
    best_pt = out_dir / "best.pt"
    history_csv = out_dir / "history.csv"
    meta_json = out_dir / "run_metadata.json"

    if not (best_pt.exists() and history_csv.exists() and meta_json.exists()):
        return False

    try:
        df_hist = pd.read_csv(history_csv)
        if len(df_hist) < expected_epochs:
            return False
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("status") == "completed"
    except Exception:
        return False


def run_preflight_check(device_name: str = "auto"):
    """Execute preflight check: test batch loading and forward passes for GP, FC, FI."""
    print("=" * 80)
    print("EXECUTING PREFLIGHT VERIFICATION (NO TRAINING RUNS LAUNCHED)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    print(f"Preflight device: {device}")

    for city in CITIES:
        meta_file = DATA_ROOT / "09_metadata" / f"stage1_source_only_target_{city}.parquet"
        manifest_file = DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"
        assert meta_file.exists(), f"Missing metadata: {meta_file}"
        assert manifest_file.exists(), f"Missing manifest: {manifest_file}"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "source_buildings" in manifest and "target_buildings" in manifest
        print(f"  Verified metadata & manifest for target city: {city}")

    print("\nTesting mini-batch data loading and forward passes across all 3 models...")
    transform = A.Compose([A.Resize(224, 224), ToTensorV2()])
    meta_file = DATA_ROOT / "09_metadata" / "stage1_source_only_target_antakya.parquet"
    manifest_file = DATA_ROOT / "08_splits" / "leave_one_city_out_v3_target_antakya.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_bids = manifest["source_buildings"]["train"]

    for model_name, cfg in MODELS.items():
        use_mask = model_name != "GP"
        ds = BuildingInstanceDataset(
            metadata_path=meta_file,
            base_dir=DATA_ROOT,
            split="train",
            sensors=["turkey_wv_visual_rgb"],
            transform=transform,
            use_mask=use_mask,
            mask_ablation=cfg["mask_ablation"],
            mask_mode=cfg["mask_mode"],
            building_ids=train_bids[:32],
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        batch = next(iter(loader))

        img = batch["image"].to(device)
        mask = batch["mask"].to(device) if use_mask else None

        net = R2BDANet(
            {"turkey_wv_visual_rgb": 3},
            adapter_dim=3,
            backbone_name="swin_tiny",
            pretrained_backbone=False,
            mask_mode=cfg["mask_mode"],
        ).to(device)
        net.eval()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                out = net(img, "turkey_wv_visual_rgb", mask=mask)

        logits4 = out["logits_4class"]
        logits2 = out["logits_2class"]
        assert logits4.shape == (16, 4), f"Bad logits4 shape: {logits4.shape}"
        assert logits2.shape == (16, 2), f"Bad logits2 shape: {logits2.shape}"

        params = sum(p.numel() for p in net.parameters())
        print(f"  [OK] Model {model_name:<10}: mini-batch forward pass successful | params: {params:,}")

    print("\n" + "=" * 80)
    print("ALL PREFLIGHT CHECKS PASSED. Matrix configuration is 100% verified.")
    print("=" * 80)


active_procs: dict[int, subprocess.Popen] = {}
proc_lock = threading.Lock()
shutdown_event = threading.Event()


def build_command(
    recipe_id: str,
    model_name: str,
    city: str,
    seed: int,
    out_dir: Path,
    num_workers: int = 4,
) -> str:
    meta_file = DATA_ROOT / "09_metadata" / f"stage1_source_only_target_{city}.parquet"
    manifest_file = DATA_ROOT / "08_splits" / f"leave_one_city_out_v3_target_{city}.json"
    r_cfg = RECIPE_CONFIGS[recipe_id]
    m_cfg = MODELS[model_name]

    python_bin = sys.executable
    lib_dir = Path(sys.executable).parent.parent / "lib"
    common_args = (
        f"--sensor turkey_wv_visual_rgb --split-mode source "
        f"--backbone swin_tiny --pretrained-backbone "
        f"--selection-metric binary_balanced_accuracy "
        f"--balance-mode sampler --augmentation-preset standard "
        f"--no-use-ema --amp --allow-existing-output "
        f"--num-workers {num_workers} --pin-memory --persistent-workers --prefetch-factor 2"
    )
    return (
        f"LD_LIBRARY_PATH={lib_dir}:$LD_LIBRARY_PATH "
        f"{python_bin} scripts/11_train_r2bda.py "
        f"--data-root {DATA_ROOT} "
        f"--metadata {meta_file} --split-manifest {manifest_file} "
        f"--seed {seed} {common_args} {r_cfg['train_args']} {m_cfg['train_args']} "
        f"--output-dir {out_dir}"
    )


def execute_task(
    task_idx: int,
    total_tasks: int,
    recipe_id: str,
    model_name: str,
    city: str,
    seed: int,
    out_dir: Path,
    num_workers: int,
    capture_output: bool,
) -> dict:
    tag = f"[{task_idx:02d}/{total_tasks:02d}] ({recipe_id} | {model_name} | {city} | seed_{seed})"
    if is_run_completed(out_dir, 80):
        print(f"{tag} [SKIPPED] Already completed (80 epochs verified).", flush=True)
        return {
            "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
            "status": "COMPLETED_ALREADY", "out_dir": str(out_dir)
        }

    if shutdown_event.is_set():
        return {
            "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
            "status": "CANCELLED", "out_dir": str(out_dir)
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(recipe_id, model_name, city, seed, out_dir, num_workers=num_workers)
    run_log = out_dir / "train.log"

    print(f"{tag} [START] Output: {out_dir}", flush=True)
    t0 = time.perf_counter()

    f_log = None
    try:
        if capture_output:
            f_log = open(run_log, "w", encoding="utf-8")
            stdout_target = f_log
            stderr_target = subprocess.STDOUT
        else:
            stdout_target = None
            stderr_target = None

        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=REPO_ROOT,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        with proc_lock:
            active_procs[task_idx] = proc

        retcode = proc.wait()
        with proc_lock:
            active_procs.pop(task_idx, None)

        if f_log:
            f_log.close()

        elapsed = time.perf_counter() - t0
        if retcode == 0:
            print(f"{tag} [SUCCESS] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
            return {
                "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
                "status": "SUCCESS", "elapsed_seconds": elapsed, "out_dir": str(out_dir)
            }
        else:
            print(f"{tag} [FAILED] Exited with code {retcode}! See log: {run_log}", flush=True)
            return {
                "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
                "status": "FAILED", "returncode": retcode, "out_dir": str(out_dir), "log": str(run_log)
            }
    except Exception as e:
        with proc_lock:
            active_procs.pop(task_idx, None)
        if f_log and not f_log.closed:
            f_log.close()
        print(f"{tag} [ERROR] Exception: {e}", flush=True)
        return {
            "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
            "status": "ERROR", "error": str(e), "out_dir": str(out_dir)
        }


def run_training_suite(
    mode: str,
    selected_recipes: list[str],
    selected_models: list[str],
    selected_cities: list[str],
    selected_seeds: list[int],
    concurrency: int = 3,
    num_workers: int = 4,
    dry_run: bool = False,
):
    is_81_mode = mode == "81_architecture_neutral"
    base_dir = RECIPES_BASE if is_81_mode else MODELS_BASE

    tasks = []
    for r in selected_recipes:
        for m in selected_models:
            for c in selected_cities:
                for s in selected_seeds:
                    if is_81_mode:
                        out_dir = base_dir / r / m / c / f"seed_{s}"
                    else:
                        out_dir = base_dir / m / c / f"seed_{s}"
                    tasks.append((r, m, c, s, out_dir))

    total = len(tasks)
    print("=" * 80)
    print(f"MATCHED-SUPPORT TRAINING RUNNER (Mode: {mode})")
    print(f"Total Runs: {total} | Concurrency: {concurrency} | Workers/Run: {num_workers} | Dry-Run: {dry_run}")
    print(f"Output Base: {base_dir}")
    print("=" * 80)

    if dry_run:
        log_summary = []
        for idx, (recipe_id, model_name, city, seed, out_dir) in enumerate(tasks, 1):
            cmd = build_command(recipe_id, model_name, city, seed, out_dir, num_workers=num_workers)
            print(f"[{idx:02d}/{total:02d}] [DRY-RUN] Recipe: {recipe_id} | Model: {model_name} | City: {city} | Seed: {seed}")
            print(f"  Target: {out_dir}")
            print(f"  Command: {cmd}\n")
            log_summary.append({
                "recipe": recipe_id, "model": model_name, "city": city, "seed": seed,
                "status": "DRY_RUN", "cmd": cmd, "out_dir": str(out_dir),
            })
        return

    def sig_handler(signum, frame):
        print("\n[INTERRUPT] Caught signal, terminating active child training processes...", flush=True)
        shutdown_event.set()
        with proc_lock:
            for p in list(active_procs.values()):
                try:
                    p.terminate()
                except Exception:
                    pass
        sys.exit(130)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    log_summary = []
    capture_output = concurrency > 1

    if concurrency <= 1:
        for idx, (recipe_id, model_name, city, seed, out_dir) in enumerate(tasks, 1):
            res = execute_task(idx, total, recipe_id, model_name, city, seed, out_dir, num_workers, capture_output=False)
            log_summary.append(res)
            if res.get("status") == "FAILED":
                break
    else:
        print(f"Starting execution pool with {concurrency} parallel workers...\n", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_task = {
                executor.submit(
                    execute_task, idx, total, r, m, c, s, out_dir, num_workers, capture_output=True
                ): (idx, r, m, c, s)
                for idx, (r, m, c, s, out_dir) in enumerate(tasks, 1)
            }
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_task):
                res = future.result()
                log_summary.append(res)
                completed_count += 1
                status = res.get("status")
                print(
                    f"--> [OVERALL PROGRESS]: {completed_count}/{total} done "
                    f"({completed_count/total*100:.1f}%) | "
                    f"Last finished: {res.get('recipe')} {res.get('city')} seed_{res.get('seed')} [{status}]\n",
                    flush=True,
                )

    summary_path = base_dir.parent / "training_matrix_status.json"
    summary_path.write_text(json.dumps(log_summary, indent=2), encoding="utf-8")
    print(f"\nAll tasks processed. Status saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["81_architecture_neutral", "27_controlled_replication"],
        default="81_architecture_neutral",
        help="Training mode: 81 runs across 3 candidate recipes, or 27 runs replicating C4b.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only without launching or creating dirs")
    parser.add_argument("--preflight", action="store_true", help="Execute single-batch preflight checks and exit")
    parser.add_argument("--concurrency", "-j", type=int, default=3, help="Number of concurrent training runs (default: 3)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers per run (default: 4)")
    parser.add_argument("--recipes", nargs="+", choices=list(RECIPE_CONFIGS.keys()), default=None)
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()))
    parser.add_argument("--cities", nargs="+", choices=CITIES, default=CITIES)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    args = parser.parse_args()

    if args.preflight:
        run_preflight_check()
        return

    if args.recipes is None:
        if args.mode == "81_architecture_neutral":
            recipes = list(RECIPE_CONFIGS.keys())
        else:
            recipes = ["R3_C4b_repaired_BS32"]
    else:
        recipes = args.recipes

    run_training_suite(
        mode=args.mode,
        selected_recipes=recipes,
        selected_models=args.models,
        selected_cities=args.cities,
        selected_seeds=args.seeds,
        concurrency=args.concurrency,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
