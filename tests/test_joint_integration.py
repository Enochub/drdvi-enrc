import numpy as np
import torch
from sklearn.datasets import make_blobs

from drdvi_enrc import DrdviRepresentation
from drdvi_enrc.clustering import ENRC


def test_drdvi_can_be_used_by_enrc():
    features, _ = make_blobs(n_samples=48, n_features=8, centers=3, random_state=1)
    features = features.astype(np.float32)
    network = DrdviRepresentation([8, 6, 4], orthogonality_weight=10.0, work_on_copy=False, random_state=1)
    model = ENRC(
        [3, 1], neural_network=network, embedding_size=4,
        pretrain_epochs=1, clustering_epochs=1, batch_size=16,
        init="random", init_subsample_size=32, final_reclustering=False,
        device=torch.device("cpu"), random_state=1,
    )
    model.fit(features)
    assert model.labels_.shape == (48, 2)

