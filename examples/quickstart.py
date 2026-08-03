"""Minimal synthetic end-to-end example; no external dataset is required."""

import numpy as np
import sys
import torch
from pathlib import Path
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drdvi_enrc import DrdviRepresentation
from drdvi_enrc.clustering import ENRC

features, _ = make_blobs(n_samples=200, n_features=16, centers=4, random_state=42)
features = features.astype(np.float32)
network = DrdviRepresentation([16, 12, 8], orthogonality_weight=10.0, work_on_copy=False, random_state=42)
model = ENRC(
    [4, 1], neural_network=network, embedding_size=8,
    pretrain_epochs=2, clustering_epochs=2, init="random",
    final_reclustering=False, device=torch.device("cpu"), random_state=42,
)
model.fit(features)
print("labels:", model.labels_.shape)
print("latent:", model.transform(features).shape)
