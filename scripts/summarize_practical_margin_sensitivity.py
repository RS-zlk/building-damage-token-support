#!/usr/bin/env python3
"""R2-BDA reproducibility script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "matched_support_v1"
    / "robustness_inference_dense_v1"
    / "robustness_summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "practical_margin_sensitivity.json"
)


def first_boundary(states, margin: float, extractor):
    for record in states.values():
        displacement = int(record["state_definition"].get("pixels", 0))
        if displacement == 0:
            continue
        point, interval = extractor(record)
        if point <= -margin and interval[1] < -margin:
            return {
                "displacement_px": displacement,
                "point_estimate_pp": point,
                "ci_95_pp": interval,
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    source = json.loads(arguments.input.read_text())
    report = {}
    for domain_name, domain in source["domains"].items():
        p1_states = domain["R1_footprint_error"]["states"]
        p2_states = domain["R2_candidate_localization_error"]["states"]
        report[domain_name] = {}
        for margin in (1.0, 2.0, 3.0):
            report[domain_name][f"{margin:.0f}_pp"] = {
                "P1_FI": first_boundary(
                    p1_states,
                    margin,
                    lambda value: (
                        value["delta_from_clean_pp"],
                        value["delta_from_clean_ci_95_pp"],
                    ),
                ),
                "P2_FC": first_boundary(
                    p2_states,
                    margin,
                    lambda value: (
                        value["models"]["FC"]["delta_from_model_clean_pp"],
                        value["models"]["FC"]["delta_from_model_clean_ci_95_pp"],
                    ),
                ),
                "P2_FI": first_boundary(
                    p2_states,
                    margin,
                    lambda value: (
                        value["models"]["FI"]["delta_from_model_clean_pp"],
                        value["models"]["FI"]["delta_from_model_clean_ci_95_pp"],
                    ),
                ),
            }
    payload = {
        "role": "post-hoc practical-margin sensitivity",
        "margins_pp": [1.0, 2.0, 3.0],
        "boundary_rule": (
            "first tested nonzero 4-pixel-grid state with point estimate at or "
            "below minus margin and 95% CI upper endpoint below minus margin"
        ),
        "domains": report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
