"""Post-hoc linear probes for information retained in a saved latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--factor-names", nargs="+", required=True)
    parser.add_argument("--factor-indices", nargs="+", type=int)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
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

    joint = np.asarray(["_".join(map(str, row)) for row in labels])
    train, test = train_test_split(np.arange(len(latent)), test_size=args.test_size,
                                   random_state=args.seed, stratify=joint)
    factors = []
    for column, name in enumerate(args.factor_names):
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed))
        probe.fit(latent[train], labels[train, column])
        factors.append({"factor": name, "accuracy": float(accuracy_score(
            labels[test, column], probe.predict(latent[test])))})
    result = {
        "analysis": "post_hoc_linear_probe", "labels_used_for_training": True,
        "labels_used_for_representation_learning": False, "seed": args.seed,
        "test_size": args.test_size, "factors": factors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
