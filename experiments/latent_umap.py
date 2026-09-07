"""Create a post-hoc UMAP plot of a saved latent, coloured by factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--factor-names", nargs="+", required=True)
    parser.add_argument("--factor-indices", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--point-size", type=float, default=2.0)
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

    coordinates = UMAP(n_components=2, n_neighbors=args.n_neighbors,
                       min_dist=args.min_dist, random_state=args.seed).fit_transform(
                           StandardScaler().fit_transform(latent))
    figure, axes = plt.subplots(1, labels.shape[1], figsize=(5 * labels.shape[1], 4), squeeze=False)
    for column, name in enumerate(args.factor_names):
        axis = axes[0, column]
        scatter = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels[:, column],
                               s=args.point_size, cmap="tab10", linewidths=0, rasterized=True)
        axis.set_title(name)
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Frozen DR-DVI latent representation")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    np.save(args.output.with_suffix(".coordinates.npy"), coordinates)


if __name__ == "__main__":
    main()
