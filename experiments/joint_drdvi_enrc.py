"""Validated unfrozen-autoencoder DRDVI + ENRC joint fine-tuning.

This is the primary joint method. It first trains (or loads) a deterministic
DRDVI autoencoder, fixes latent standardization statistics, initializes ENRC
with NrKmeans, then updates encoder, decoder, and ENRC together.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drdvi_enrc import DrdviAutoencoder
from drdvi_enrc.clustering import _ENRC_Module, enrc_init
from drdvi_enrc.utils import load_dataset, load_yaml, matched_score_rows, set_seed


class StandardizedDrdviAutoencoder(torch.nn.Module):
    def __init__(self, base: DrdviAutoencoder, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.base = base
        self.register_buffer("latent_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("latent_std", torch.from_numpy(std.astype(np.float32)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (self.base.encode(x) - self.latent_mean) / (self.latent_std + 1e-8)

    def decode(self, standardized_z: torch.Tensor) -> torch.Tensor:
        return self.base.decode(standardized_z * (self.latent_std + 1e-8) + self.latent_mean)


def encode_numpy(model, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    values = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            values.append(model.encode(batch.to(device)).cpu().numpy())
    return np.vstack(values).astype(np.float32)


def pretrain_autoencoder(features: np.ndarray, layers: list[int], epochs: int, batch_size: int,
                         learning_rate: float, seed: int, device: torch.device) -> DrdviAutoencoder:
    model = DrdviAutoencoder(layers, work_on_copy=False, random_state=seed).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in tqdm.trange(epochs, desc="DRDVI AE-only pretrain"):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            loss = torch.nn.functional.mse_loss(model.decode(model.encode(batch)), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def load_or_pretrain(args, config, features, device) -> DrdviAutoencoder:
    layers = [features.shape[1], *config["hidden_dims"], int(config["embedding_dim"])]
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = DrdviAutoencoder(checkpoint["layers"], work_on_copy=False, random_state=args.seed).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    pretrain_epochs = args.quick_epochs or args.pretrain_epochs
    model = pretrain_autoencoder(
        features, layers, pretrain_epochs, int(config["batch_size"]),
        args.pretrain_lr, args.seed, device,
    )
    checkpoint = Path(args.out_dir) / "ae_pretrain.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "layers": layers}, checkpoint)
    return model


def train(args: argparse.Namespace) -> list[dict]:
    config = load_yaml(args.config)
    dataset_name = config["dataset"]
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, labels, label_names = load_dataset(dataset_name, args.data_root, config, args.seed)
    features = features.astype(np.float32)
    batch_size = int(config["batch_size"])
    base = load_or_pretrain(args, config, features, device)

    initial_raw = encode_numpy(base, features, batch_size, device)
    model = StandardizedDrdviAutoencoder(base, initial_raw.mean(0), initial_raw.std(0) + 1e-8).to(device)
    initial_z = encode_numpy(model, features, batch_size, device)
    n_clusters = [int(np.unique(labels[:, index]).size) for index in range(labels.shape[1])]
    centers, projections, rotation, beta_weights = enrc_init(
        data=initial_z, n_clusters=n_clusters, device=device, init="nrkmeans",
        rounds=args.init_rounds, epochs=args.init_epochs, batch_size=batch_size,
        max_iter=args.init_max_iter, optimizer_params={"lr": args.enrc_lr, "betas": (0.9, 0.99)},
        optimizer_class=torch.optim.Adam, random_state=np.random.RandomState(args.seed), debug=args.debug,
    )
    module = _ENRC_Module(
        centers, projections, rotation, beta_weights=beta_weights,
        clustering_loss_weight=args.clustering_loss_weight, ssl_loss_weight=args.lambda_rec,
    ).to_device(device)

    encoder = [base.a, *base.B.parameters(), *base.encoder_a.parameters(), *base.encoder_b.parameters()]
    decoder = [*base.A.parameters(), base.Az]
    optimizer = torch.optim.Adam([
        {"params": encoder, "lr": args.encoder_lr, "betas": (0.9, 0.99)},
        {"params": decoder, "lr": args.encoder_lr, "betas": (0.9, 0.99)},
        {"params": [module.V], "lr": args.enrc_lr, "betas": (0.9, 0.99)},
        {"params": [module.beta_weights], "lr": args.enrc_lr * 10, "betas": (0.9, 0.99)},
    ])
    loader = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    epochs = args.epochs or args.quick_epochs or 500
    history = []
    for epoch in tqdm.trange(epochs, desc="Unfrozen AE joint DRDVI+ENRC"):
        totals = {"total_loss": 0.0, "cluster_loss": 0.0, "reconstruction_loss": 0.0}
        batches = 0
        model.train(); module.train()
        for (batch,) in loader:
            batch = batch.to(device)
            latent = model.encode(batch)
            cluster_loss, rotated, rotated_back, assignments = module(latent)
            reconstruction_loss = torch.nn.functional.mse_loss(model.decode(rotated_back), batch)
            loss = args.clustering_loss_weight * cluster_loss + args.lambda_rec * reconstruction_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            with torch.no_grad():
                module.update_centers(rotated, assignments)
            totals["total_loss"] += float(loss.detach())
            totals["cluster_loss"] += float(cluster_loss.detach())
            totals["reconstruction_loss"] += float(reconstruction_loss.detach())
            batches += 1
        history.append({"epoch": epoch, **{key: value / max(batches, 1) for key, value in totals.items()}})

    module.P = module.get_P()
    evaluation = DataLoader(TensorDataset(torch.from_numpy(features)), batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval(); module.eval()
    with torch.no_grad():
        for (batch,) in evaluation:
            predictions.append(module.predict(model.encode(batch.to(device)), use_P=True))
    predictions = np.vstack(predictions)
    rows = [{"dataset": dataset_name, **row} for row in matched_score_rows(
        "joint_drdvi_enrc", labels, predictions, label_names, True
    )]
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("summary_scores.csv", rows), ("loss_history.csv", history)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    diagnostics = {"method": "unfrozen_ae_manual_joint", "epochs": epochs,
                   "encoder_lr": args.encoder_lr, "enrc_lr": args.enrc_lr,
                   "lambda_rec": args.lambda_rec, "clustering_loss_weight": args.clustering_loss_weight,
                   "P": [list(map(int, values)) for values in module.P]}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained DRDVI AE checkpoint.")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick-epochs", type=int, default=None)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--lambda-rec", type=float, default=0.1)
    parser.add_argument("--enrc-lr", type=float, default=1e-4)
    parser.add_argument("--clustering-loss-weight", type=float, default=0.1)
    parser.add_argument("--init-rounds", type=int, default=10)
    parser.add_argument("--init-epochs", type=int, default=10)
    parser.add_argument("--init-max-iter", type=int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
