"""Construct the fixed prototype-distance bridge used for NR-Objects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drdvi_enrc.utils.metrics import matched_score_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--factor-names", nargs="+", required=True)
    parser.add_argument("--clusters", nargs="+", type=int, required=True)
    parser.add_argument("--factor-indices", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
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

    scaler = StandardScaler().fit(latent)
    standardized = scaler.transform(latent)
    models = [KMeans(count, n_init=20, random_state=args.seed).fit(standardized)
              for count in args.clusters]
    pseudo = np.column_stack([model.labels_ for model in models])
    raw_blocks = [-model.transform(standardized) ** 2 for model in models]
    block_means = [block.mean(0) for block in raw_blocks]
    block_stds = [np.maximum(block.std(0), 1e-6) for block in raw_blocks]
    bridge = np.concatenate([(block - mean) / std
                             for block, mean, std in zip(raw_blocks, block_means, block_stds)], axis=1)

    rows = matched_score_rows(
        "independent_kmeans_initialization", labels, pseudo, args.factor_names, True
    )
    result = {
        "protocol": "fixed_prototype_distance_bridge_initialization_v1",
        "labels_used_for_fitting": False, "labels_used_for_post_hoc_evaluation": True,
        "seed": args.seed, "cluster_counts": args.clusters,
        "bridge_dimensions": int(bridge.shape[1]), "metrics": rows,
        "mean_acc": float(np.mean([row["ACC"] for row in rows])),
        "mean_nmi": float(np.mean([row["NMI"] for row in rows])),
        "mean_ari": float(np.mean([row["ARI"] for row in rows])),
        "note": "This constructs the fixed bridge and pseudo-label initialization; it is not standard joint DR-DVI--ENRC training.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "bridge.npy", bridge.astype(np.float32))
    np.save(args.output_dir / "pseudo_labels.npy", pseudo)
    np.savez(args.output_dir / "bridge_state.npz", latent_mean=scaler.mean_, latent_scale=scaler.scale_,
             **{f"centers_{count}": model.cluster_centers_ for count, model in zip(args.clusters, models)},
             **{f"block_mean_{i}": value for i, value in enumerate(block_means)},
             **{f"block_std_{i}": value for i, value in enumerate(block_stds)})
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
