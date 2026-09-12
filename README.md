# R2-BDA reproducibility code

Code to reproduce the training, frozen evaluation, and inference-only robustness analyses reported in the R2-BDA study.

## Scope

This repository contains implementation code and the public study protocol. It intentionally excludes imagery, footprint masks, split manifests, checkpoints, prediction tables, and generated results. Obtain redistributable data and accompanying metadata from the associated Zenodo record, then place or link them under `data/`. Do not commit these files.

The three Stage-3 spatial-token support rules share the same Swin-Tiny encoder, dual-branch readout, heads, training procedure, and 27,981,697 parameters:

- **GP** (global pooling): mean pooling over all 196 tokens in the 14 x 14 feature map.
- **FC** (fixed-center support): four fixed central interior tokens and twelve context tokens. It does not use a footprint after a candidate patch exists.
- **FI** (footprint-informed support): four highest-coverage interior tokens and twelve nearest zero-coverage context tokens. Distances are from token centers to the nearest footprint pixel.

Footprints are used only after encoding to construct FI support; they are not an encoder input or an attention mask.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10 or later. The scripts expect a project-local `data/` directory. You may instead set `R2_BDA_DATA_DIR` to the extracted Zenodo dataset directory where supported. Dataset-specific metadata and split manifests should retain the layout supplied with the Zenodo release.

## Main entry points

- `scripts/run_matched_support_training.py`: matched training runs.
- `scripts/select_architecture_neutral_recipe.py`: common recipe selection.
- `scripts/select_unified_global_threshold.py`: frozen common threshold selection.
- `scripts/run_matched_support_primary_evaluation.py`: Turkey leave-one-city-out evaluation and paired bootstrap.
- `scripts/run_matched_support_robustness_inference.py`: P1 and P2 inference-only perturbation analyses.
- `scripts/run_method_specific_threshold_sensitivity.py`, `scripts/run_spatial_block_bootstrap_sensitivity.py`, and `scripts/summarize_practical_margin_sensitivity.py`: post-hoc sensitivity analyses.

The spatial-block bootstrap additionally requires the released `09_metadata/turkey_building_coordinates_utm37n.csv` table (`building_id`, `easting_m`, and `northing_m`). It does not require the source shapefiles.

Generated outputs are written to ignored result directories. Before public release, set the Zenodo DOI and choose a license for this repository.
