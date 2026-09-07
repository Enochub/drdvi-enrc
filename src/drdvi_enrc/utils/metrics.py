"""Metrics for matching multiple predicted clusterings to ground truth factors."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted, true = np.unique(y_pred), np.unique(y_true)
    counts = np.zeros((len(predicted), len(true)), dtype=np.int64)
    pred_index = {value: index for index, value in enumerate(predicted)}
    true_index = {value: index for index, value in enumerate(true)}
    for pred_value, true_value in zip(y_pred, y_true):
        counts[pred_index[pred_value], true_index[true_value]] += 1
    rows, columns = linear_sum_assignment(counts.max() - counts)
    return float(counts[rows, columns].sum() / len(y_true))


def matched_score_rows(method: str, labels: np.ndarray, predictions: np.ndarray,
                       label_names: list[str],
                       require_equal_cluster_counts: bool = False) -> list[dict]:
    if labels.ndim == 1:
        labels = labels[:, None]
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    nmi = np.zeros((labels.shape[1], predictions.shape[1]))
    for true_idx in range(labels.shape[1]):
        for pred_idx in range(predictions.shape[1]):
            nmi[true_idx, pred_idx] = normalized_mutual_info_score(
                labels[:, true_idx], predictions[:, pred_idx]
            )
    matching_scores = nmi.copy()
    if require_equal_cluster_counts:
        true_counts = [np.unique(labels[:, index]).size for index in range(labels.shape[1])]
        predicted_counts = [np.unique(predictions[:, index]).size for index in range(predictions.shape[1])]
        for true_idx, true_count in enumerate(true_counts):
            for pred_idx, predicted_count in enumerate(predicted_counts):
                if true_count != predicted_count:
                    matching_scores[true_idx, pred_idx] = -1.0
    true_indices, pred_indices = linear_sum_assignment(-matching_scores)
    if require_equal_cluster_counts and any(
        matching_scores[true_idx, pred_idx] < 0
        for true_idx, pred_idx in zip(true_indices, pred_indices)
    ):
        raise ValueError("No complete view matching with equal cluster counts")
    rows = []
    for true_idx, pred_idx in zip(true_indices, pred_indices):
        truth, prediction = labels[:, true_idx], predictions[:, pred_idx]
        rows.append({
            "method": method,
            "label": label_names[true_idx],
            "clustering": int(pred_idx),
            "n_clusters": int(np.unique(prediction).size),
            "ACC": clustering_accuracy(truth, prediction),
            "NMI": float(nmi[true_idx, pred_idx]),
            "ARI": float(adjusted_rand_score(truth, prediction)),
        })
    return rows
