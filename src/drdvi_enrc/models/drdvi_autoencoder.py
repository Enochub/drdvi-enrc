import math

import numpy as np
import torch

from clustpy.deep.neural_networks._abstract_autoencoder import _AbstractAutoencoder


class DrdviAutoencoder(_AbstractAutoencoder):
    """Autoencoder exposing a deterministic DRDVI-style encoder to ClustPy.

    The encoder applies the DRDVI inference matrices and maps the final hidden
    representation to the mean of a Beta distribution. The decoder mirrors the
    generative matrix path so the network can be pretrained and jointly trained
    by deep clustering algorithms such as ENRC.

    Parameters
    ----------
    layers : list
        Layer sizes from input to embedding, for example ``[784, 256, 64, 16]``.
    work_on_copy : bool
        If True, deep clustering algorithms optimize a copy of this network.
    random_state : np.random.RandomState | int
        Random state used by the inherited ClustPy training routine.
    """

    def __init__(
        self,
        layers: list,
        work_on_copy: bool = True,
        random_state: np.random.RandomState | int = None,
    ):
        super().__init__(work_on_copy=work_on_copy, random_state=random_state)
        if len(layers) < 3:
            raise ValueError("layers must contain input, at least one hidden, and embedding dimensions.")
        if any(not isinstance(dim, int) or dim <= 0 for dim in layers):
            raise ValueError("All entries in layers must be positive integers.")

        self.layers = layers.copy()
        hidden_dims = self.layers[1:-1]
        embedding_dim = self.layers[-1]

        self.a = torch.nn.Parameter(torch.linspace(-1.0, 1.0, len(hidden_dims)))
        self.B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(self._init_weight(out_dim, in_dim))
                for in_dim, out_dim in zip(self.layers[:-2], hidden_dims)
            ]
        )
        self.A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(self._init_weight(in_dim, out_dim))
                for in_dim, out_dim in zip(self.layers[:-2], hidden_dims)
            ]
        )

        self.Az = torch.nn.Parameter(self._init_weight(hidden_dims[-1], embedding_dim))
        self.encoder_a = torch.nn.Linear(hidden_dims[-1], embedding_dim, bias=False)
        self.encoder_b = torch.nn.Linear(hidden_dims[-1], embedding_dim, bias=False)

    @staticmethod
    def _init_weight(rows: int, cols: int) -> torch.Tensor:
        bound = math.sqrt(1.0 / cols)
        return torch.empty(rows, cols).uniform_(-bound, bound)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic DRDVI inference path."""
        h = x.reshape(1, -1) if x.ndim == 1 else x
        if h.shape[1] != self.layers[0]:
            raise ValueError(
                f"Input layer of the encoder ({self.layers[0]}) does not match input sample ({h.shape[1]})."
            )
        for layer_idx, b_matrix in enumerate(self.B):
            h = torch.sqrt(torch.sigmoid(self.a[layer_idx])) * torch.matmul(h, b_matrix.t())
        return h

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the alpha and beta parameters of the latent Beta distributions."""
        h = self.encode_hidden(x)
        alpha = torch.exp(self.encoder_a(h).clamp(min=-20.0, max=20.0))
        beta = torch.exp(self.encoder_b(h).clamp(min=-20.0, max=20.0))
        return alpha, beta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deterministic latent representation E[z] = alpha / (alpha + beta)."""
        single_sample = x.ndim == 1
        alpha, beta = self.encode_params(x)
        embedded = alpha / (alpha + beta)
        return embedded[0] if single_sample else embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """Reconstruct samples through the mirrored DRDVI generative path."""
        single_sample = embedded.ndim == 1
        h = embedded.reshape(1, -1) if single_sample else embedded
        if h.shape[1] != self.layers[-1]:
            raise ValueError(
                f"Input layer of the decoder ({self.layers[-1]}) does not match embedding ({h.shape[1]})."
            )
        h = torch.matmul(h, self.Az.t())
        for layer_idx, a_matrix in reversed(list(enumerate(self.A))):
            h = torch.matmul(h, a_matrix.t())
            if layer_idx != 0:
                h = torch.relu(h)
        return h[0] if single_sample else h

    def orthogonality_penalty(self) -> torch.Tensor:
        """Measure deviations of the DRDVI inference matrices from row orthogonality."""
        penalty = torch.zeros((), device=self.a.device, dtype=self.a.dtype)
        for b_matrix in self.B:
            eye = torch.eye(b_matrix.shape[0], device=b_matrix.device, dtype=b_matrix.dtype)
            penalty = penalty + (torch.matmul(b_matrix, b_matrix.t()) - eye).pow(2.0).sum()
        return penalty
