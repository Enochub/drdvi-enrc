"""Experimental full stochastic-DRDVI-objective + ENRC training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drdvi_enrc import DrdviRepresentation
from drdvi_enrc.clustering import ENRC
from drdvi_enrc.utils import load_dataset, load_yaml, matched_score_rows, set_seed


def run(args: argparse.Namespace) -> list[dict]:
    config = load_yaml(args.config)
    name = config["dataset"]
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, labels, label_names = load_dataset(name, args.data_root, config, args.seed)
    features = features.astype(np.float32)
    n_clusters = [int(np.unique(labels[:, index]).size) for index in range(labels.shape[1])]
    if len(n_clusters) == 1:
        n_clusters.append(1)
    network = DrdviRepresentation(
        [features.shape[1], *config["hidden_dims"], int(config["embedding_dim"])],
        orthogonality_weight=args.orthogonality_weight,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        work_on_copy=False,
        random_state=args.seed,
    )
    model = ENRC(
        n_clusters=n_clusters,
        neural_network=network,
        embedding_size=int(config["embedding_dim"]),
        batch_size=int(config["batch_size"]),
        pretrain_epochs=args.epochs or args.quick_epochs or int(config["drdvi_epochs"]),
        clustering_epochs=args.epochs or args.quick_epochs or int(config["enrc_clustering_epochs"]),
        pretrain_optimizer_params={"lr": args.drdvi_lr},
        clustering_optimizer_params={"lr": args.enrc_lr},
        init="random", final_reclustering=False, device=device,
        random_state=args.seed, debug=args.debug,
    )
    model.fit(features)
    rows = matched_score_rows("joint_full_drdvi_enrc", labels, model.labels_, label_names)
    rows = [{"dataset": name, **row} for row in rows]
    output = Path(args.out_dir) / "summary_scores.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override pretraining and clustering epoch counts.")
    parser.add_argument("--quick-epochs", type=int, default=None)
    parser.add_argument("--drdvi-lr", type=float, default=1e-3)
    parser.add_argument("--enrc-lr", type=float, default=1e-4)
    parser.add_argument("--orthogonality-weight", type=float, default=100.0)
    parser.add_argument("--variance-weight", type=float, default=0.0)
    parser.add_argument("--covariance-weight", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
