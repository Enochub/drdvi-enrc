"""Run ENRC, DRDVI+KMeans, and frozen DRDVI+ENRC baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import encode_batchwise
from drdvi_enrc import DrdviRepresentation
from drdvi_enrc.clustering import ENRC
from drdvi_enrc.utils import load_dataset, load_yaml, matched_score_rows, set_seed


class IdentityAutoencoder(torch.nn.Module):
    """Minimal ClustPy-compatible network for clustering fixed representations."""

    def __init__(self):
        super().__init__()
        self.fitted, self.work_on_copy, self.allow_nd_input = True, False, False
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def forward(self, x):
        return x

    def loss(self, batch, ssl_loss_fn=None, device=torch.device("cpu"), corruption_fn=None):
        data = batch[1].to(device)
        return self.dummy.to(device) * 0.0, data, data


def _clusters(labels: np.ndarray) -> list[int]:
    values = [int(np.unique(labels[:, index]).size) for index in range(labels.shape[1])]
    return values if len(values) > 1 else values + [1]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict]:
    config = load_yaml(args.config)
    name = config["dataset"]
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, labels, label_names = load_dataset(name, args.data_root, config, args.seed)
    features = features.astype(np.float32)
    n_clusters = _clusters(labels)
    rows: list[dict] = []

    if "enrc" in args.methods:
        model = ENRC(
            n_clusters=n_clusters,
            embedding_size=int(config["embedding_dim"]),
            batch_size=int(config["batch_size"]),
            pretrain_epochs=args.epochs or args.quick_epochs or int(config["enrc_pretrain_epochs"]),
            clustering_epochs=args.epochs or args.quick_epochs or int(config["enrc_clustering_epochs"]),
            init="random", final_reclustering=False, device=device, random_state=args.seed,
        )
        model.fit(features)
        rows.extend(matched_score_rows("enrc", labels, model.labels_, label_names))

    if {"drdvi", "frozen"}.intersection(args.methods):
        layers = [features.shape[1], *config["hidden_dims"], int(config["embedding_dim"])]
        representation = DrdviRepresentation(layers, work_on_copy=False, random_state=args.seed).to(device)
        loader = get_dataloader(features, int(config["batch_size"]), shuffle=True, drop_last=True)
        epochs = args.epochs or args.quick_epochs or int(config["drdvi_epochs"])
        representation.fit(n_epochs=epochs, dataloader=loader, optimizer_params={"lr": 1e-3})
        evaluation_loader = get_dataloader(features, int(config["batch_size"]), shuffle=False, drop_last=False)
        latent = encode_batchwise(evaluation_loader, representation).astype(np.float32)
        standardized = ((latent - latent.mean(0)) / (latent.std(0) + 1e-8)).astype(np.float32)

        if "drdvi" in args.methods:
            predictions = np.column_stack([
                KMeans(int(np.unique(labels[:, index]).size), n_init=20, random_state=args.seed + index)
                .fit_predict(standardized)
                for index in range(labels.shape[1])
            ])
            rows.extend(matched_score_rows("drdvi_kmeans", labels, predictions, label_names))

        if "frozen" in args.methods:
            frozen = ENRC(
                n_clusters=n_clusters, embedding_size=standardized.shape[1],
                neural_network=IdentityAutoencoder(), ssl_loss_weight=0.0,
                batch_size=int(config["batch_size"]), pretrain_epochs=0,
                clustering_epochs=args.epochs or args.quick_epochs or int(config["enrc_clustering_epochs"]),
                init="random", final_reclustering=False, device=device, random_state=args.seed,
            )
            frozen.fit(standardized)
            rows.extend(matched_score_rows("frozen_drdvi_enrc", labels, frozen.labels_, label_names))

    for row in rows:
        row["dataset"] = name
    ordered = [{"dataset": row.pop("dataset"), **row} for row in rows]
    _write(Path(args.out_dir) / "summary_scores.csv", ordered)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=["enrc", "drdvi", "frozen"], default=["enrc", "drdvi", "frozen"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override all training stages with this epoch count.")
    parser.add_argument("--quick-epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
