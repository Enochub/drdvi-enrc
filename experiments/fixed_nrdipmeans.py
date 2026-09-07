"""Run the official Scala Nr-DipMeans implementation on a fixed latent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drdvi_enrc.utils.metrics import clustering_accuracy  # noqa: E402


def read_summary(path):
    return dict(line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scala-project", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--factor-names", nargs="+", required=True)
    parser.add_argument("--factor-indices", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--significance", type=float, default=0.001)
    parser.add_argument("--max-total-clusters", type=int, default=20)
    parser.add_argument("--sbt", default="sbt")
    args = parser.parse_args()

    latent = np.asarray(np.load(args.latent), dtype=np.float64)
    labels = np.asarray(np.load(args.labels))
    if labels.ndim == 1:
        labels = labels[:, None]
    indices = args.factor_indices or list(range(len(args.factor_names)))
    labels = labels[:, indices]
    if latent.ndim != 2 or len(latent) != len(labels):
        raise ValueError(f"Incompatible shapes: latent={latent.shape}, labels={labels.shape}")
    if labels.shape[1] != len(args.factor_names):
        raise ValueError("factor names and selected label columns must have equal lengths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    standardized = (latent - latent.mean(0, keepdims=True)) / np.maximum(latent.std(0, keepdims=True), 1e-12)
    input_csv = args.output_dir / "latent_standardized.csv"
    np.savetxt(input_csv, standardized, delimiter=";", fmt="%.10g")
    command = " ".join([
        "runMain RunDrdviNrDipmeans", str(input_csv.resolve()),
        str(args.output_dir.resolve()), str(len(args.factor_names)), str(args.seed),
        str(args.n_runs), str(args.significance), str(args.max_total_clusters),
    ])
    subprocess.run([args.sbt, command], cwd=args.scala_project, check=True)

    predictions = np.loadtxt(args.output_dir / "predictions.csv", delimiter=";", dtype=np.int64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    nmi = np.asarray([[normalized_mutual_info_score(labels[:, i], predictions[:, j])
                       for j in range(predictions.shape[1])]
                      for i in range(labels.shape[1])])
    truth_spaces, predicted_spaces = linear_sum_assignment(-nmi)
    summary = read_summary(args.output_dir / "model_summary.tsv")
    counts = [int(value) for value in summary["cluster_counts"].split(",")]
    metrics = []
    for i, j in zip(truth_spaces, predicted_spaces):
        metrics.append({
            "truth_space": int(i), "truth_name": args.factor_names[i], "predicted_space": int(j),
            "truth_clusters": int(np.unique(labels[:, i]).size), "predicted_clusters": counts[j],
            "acc": clustering_accuracy(labels[:, i], predictions[:, j]), "nmi": float(nmi[i, j]),
            "ari": float(adjusted_rand_score(labels[:, i], predictions[:, j])),
        })
    metrics.sort(key=lambda row: row["truth_space"])
    result = {
        "protocol": "fixed_standardized_drdvi_latent_official_scala_nrdipmeans_v1",
        "seed": args.seed, "n_runs": args.n_runs, "significance": args.significance,
        "labels_used_for_fitting": False, "run_selection": "minimum_unsupervised_NrKMeans_cost",
        "space_matching": "hungarian_max_nmi_for_evaluation_only",
        "cluster_counts": counts,
        "subspace_dimensions": [int(value) for value in summary["subspace_dimensions"].split(",")],
        "best_cost": float(summary["best_cost"]), "best_seed": int(summary["best_seed"]),
        "nmi_matrix": nmi.tolist(), "metrics": metrics,
        "mean_acc": float(np.mean([row["acc"] for row in metrics])),
        "mean_nmi": float(np.mean([row["nmi"] for row in metrics])),
        "mean_ari": float(np.mean([row["ari"] for row in metrics])),
    }
    np.save(args.output_dir / "predictions.npy", predictions)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
