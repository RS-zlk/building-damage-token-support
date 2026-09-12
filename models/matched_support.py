"""Stage-3 token-support construction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_fc_support_14x14() -> np.ndarray:
    support = np.zeros((2, 14, 14), dtype=np.float32)
    support[0, 6:8, 6:8] = 1.0
    support[1, 5:9, 5:9] = 1.0
    support[1, 6:8, 6:8] = 0.0
    return support


def compute_fi_support_14x14(mask_224: np.ndarray) -> np.ndarray:
    num_positive_pixels = int((mask_224 > 0).sum())
    if num_positive_pixels == 0:
        raise ValueError(
            "Received an empty footprint mask (0 positive pixels). "
            "Empty masks indicate a data/pipeline fault and must not be silently infilled."
        )

    support = np.zeros((2, 14, 14), dtype=np.float32)

    coverage = mask_224.reshape(14, 16, 14, 16).mean(axis=(1, 3))
    cov_flat = coverage.ravel()  # shape (196,)

    covered_indices = np.where(cov_flat > 0)[0]
    if len(covered_indices) >= 4:
        order = np.lexsort((np.arange(196), -cov_flat))
        interior_indices = order[:4]
    else:
        ys, xs = np.where(mask_224 > 0)
        cy_cell = float(ys.mean()) / 16.0
        cx_cell = float(xs.mean()) / 16.0

        grid_r, grid_c = np.indices((14, 14))
        dist_to_center = (grid_r + 0.5 - cy_cell) ** 2 + (grid_c + 0.5 - cx_cell) ** 2
        dist_flat = dist_to_center.ravel()

        int_list: list[int] = []
        if len(covered_indices) > 0:
            cov_sorted = covered_indices[np.lexsort((covered_indices, -cov_flat[covered_indices]))]
            int_list.extend(cov_sorted.tolist())

        remaining_order = np.lexsort((np.arange(196), dist_flat))
        for idx in remaining_order:
            if idx not in int_list:
                int_list.append(int(idx))
            if len(int_list) == 4:
                break
        interior_indices = np.array(int_list[:4], dtype=int)

    for idx in interior_indices:
        r, c = divmod(idx, 14)
        support[0, r, c] = 1.0

    exterior_candidates = np.where((cov_flat == 0) & (support[0].ravel() == 0))[0]

    dist_map = distance_transform_edt(1.0 - (mask_224 > 0).astype(np.float32))
    cell_dists = dist_map[8::16, 8::16].ravel()

    ext_dists = cell_dists[exterior_candidates]
    ext_order = np.lexsort((exterior_candidates, ext_dists))
    selected_context = exterior_candidates[ext_order[:12]]

    for idx in selected_context:
        r, c = divmod(idx, 14)
        support[1, r, c] = 1.0

    return support
