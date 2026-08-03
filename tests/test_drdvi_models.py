import torch

from drdvi_enrc import DrdviAutoencoder, DrdviRepresentation


def test_autoencoder_shapes_and_gradient():
    model = DrdviAutoencoder([8, 6, 3], work_on_copy=False, random_state=1)
    data = torch.rand(10, 8)
    latent = model.encode(data)
    reconstructed = model.decode(latent)
    assert latent.shape == (10, 3)
    assert reconstructed.shape == data.shape
    torch.nn.functional.mse_loss(reconstructed, data).backward()


def test_full_drdvi_objective_is_finite():
    model = DrdviRepresentation([8, 6, 3], orthogonality_weight=10.0, work_on_copy=False, random_state=1)
    loss = model.enrc_loss(torch.rand(10, 8))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

