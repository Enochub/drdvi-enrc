"""Run NrKMeans on a saved, fixed DR-DVI representation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from clustpy.alternative import NrKmeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drdvi_enrc.utils.metrics import clustering_accuracy  # noqa: E402


def evaluate(labels, predictions, names, require_equal_counts):
    nmi = np.asarray([[normalized_mutual_info_score(labels[:, i], predictions[:, j])
                       for j in range(predictions.shape[1])]
                      for i in range(labels.shape[1])])
    truth_counts = [np.unique(labels[:, i]).size for i in range(labels.shape[1])]
    pred_counts = [np.unique(predictions[:, j]).size for j in range(predictions.shape[1])]
    score = nmi.copy()
    if require_equal_counts:
        for i, truth_count in enumerate(truth_counts):
            for j, pred_count in enumerate(pred_counts):
                if truth_count != pred_count:
                    score[i, j] = -1.0
    rows, columns = linear_sum_assignment(-score)
    if require_equal_counts and any(score[i, j] < 0 for i, j in zip(rows, columns)):
        raise ValueError("No complete one-to-one matching with equal cluster counts")
    metrics = []
    for i, j in zip(rows, columns):
        metrics.append({
            "truth_space": int(i), "truth_name": names[i], "predicted_space": int(j),
            "truth_clusters": int(truth_counts[i]), "predicted_clusters": int(pred_counts[j]),
            "acc": clustering_accuracy(labels[:, i], predictions[:, j]),
            "nmi": float(nmi[i, j]),
            "ari": float(adjusted_rand_score(labels[:, i], predictions[:, j])),
        })
    metrics.sort(key=lambda row: row["truth_space"])
    return {
        "matching": "maximum_nmi_equal_cluster_count" if require_equal_counts else "maximum_nmi",
        "nmi_matrix": nmi.tolist(), "metrics": metrics,
        "mean_acc": float(np.mean([row["acc"] for row in metrics])),
        "mean_nmi": float(np.mean([row["nmi"] for row in metrics])),
        "mean_ari": float(np.mean([row["ari"] for row in metrics])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--factor-names", nargs="+", required=True)
    parser.add_argument("--clusters", nargs="+", type=int, required=True)
    parser.add_argument("--factor-indices", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--allow-cross-cardinality-matching", action="store_true")
    args = parser.parse_args()

    latent = np.asarray(np.load(args.latent), dtype=np.float64)
    labels = np.asarray(np.load(args.labels))
    if labels.ndim == 1:
        labels = labels[:, None]
    indices = args.factor_indices or list(range(len(args.factor_names)))
    labels = labels[:, indices]
    if latent.ndim != 2 or len(latent) != len(labels):
        raise ValueError(f"Incompatible shapes: latent={latent.shape}, labels={labels.shape}")
    if len(args.clusters) != len(args.factor_names) or labels.shape[1] != len(args.factor_names):
        raise ValueError("clusters, factor names, and selected label columns must have equal lengths")

    standardized = StandardScaler().fit_transform(latent)
    model = NrKmeans(n_clusters=args.clusters, n_init=args.n_init,
                     max_iter=args.max_iter, cost_type="default",
                     random_state=args.seed).fit(standardized)
    predictions = np.asarray(model.labels_)
    result = {
        "protocol": "fixed_standardized_drdvi_latent_nrkmeans_v1",
        "seed": args.seed, "labels_used_for_fitting": False,
        "requested_clusters": args.clusters,
        "final_clusters": [int(value) for value in model.n_clusters_final_],
        "subspace_dimensions": [int(value) for value in model.m_],
        "projections": [[int(value) for value in projection] for projection in model.P_],
        **evaluate(labels, predictions, args.factor_names,
                   not args.allow_cross_cardinality_matching),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "predictions.npy", predictions)
    np.save(args.output_dir / "rotation.npy", np.asarray(model.V_))
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
