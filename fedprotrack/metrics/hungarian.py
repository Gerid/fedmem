from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_predictions(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Align predicted concept IDs to ground-truth IDs via the Hungarian algorithm.

    Finds the permutation of predicted concept labels that maximises the
    overlap (agreement count) with the ground-truth labels.  When the number
    of predicted concepts exceeds the number of ground-truth concepts the
    confusion matrix is rectangular and the surplus predicted concepts are
    mapped to ``-1``.

    Parameters
    ----------
    ground_truth : np.ndarray
        Shape (K, T), dtype int.  Ground-truth concept IDs.
    predicted : np.ndarray
        Shape (K, T), dtype int.  Predicted concept IDs (may use arbitrary
        non-negative integer labels).

    Returns
    -------
    aligned_predictions : np.ndarray
        Shape (K, T), dtype int32.  Predicted concept IDs remapped to
        ground-truth label space; unmatched predicted concepts become ``-1``.
    mapping : dict[int, int]
        Mapping from original predicted concept ID to ground-truth concept ID.
        Unmatched predicted IDs map to ``-1``.

    Notes
    -----
    The implementation builds a confusion matrix C of shape
    ``(n_pred_concepts, n_gt_concepts)`` and solves
    ``linear_sum_assignment(-C)`` to maximise agreement.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)

    gt_concepts = np.unique(ground_truth)   # sorted unique gt IDs
    pred_concepts = np.unique(predicted)    # sorted unique pred IDs

    n_pred = len(pred_concepts)
    n_gt = len(gt_concepts)

    # Build confusion matrix C[i, j] = # cells where pred==pred_concepts[i]
    # AND gt==gt_concepts[j].
    C = np.zeros((n_pred, n_gt), dtype=np.int64)
    for i, pc in enumerate(pred_concepts):
        mask_pred = predicted == pc
        for j, gc in enumerate(gt_concepts):
            C[i, j] = int(np.sum(mask_pred & (ground_truth == gc)))

    # Hungarian: minimise -C  ⟺  maximise C.
    row_ind, col_ind = linear_sum_assignment(-C)

    # Build mapping: pred_concept → gt_concept.
    # pred concepts not included in row_ind are unmatched → -1.
    mapping: dict[int, int] = {int(pc): -1 for pc in pred_concepts}
    for r, c in zip(row_ind, col_ind):
        mapping[int(pred_concepts[r])] = int(gt_concepts[c])

    # Apply mapping to the full prediction array.
    aligned = np.full_like(predicted, fill_value=-1, dtype=np.int32)
    for pred_id, gt_id in mapping.items():
        aligned[predicted == pred_id] = gt_id

    return aligned, mapping
