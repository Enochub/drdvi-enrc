import math
from collections.abc import Callable

import numpy as np
import torch

from clustpy.deep._utils import mean_squared_error
from clustpy.deep.neural_networks._abstract_autoencoder import _AbstractAutoencoder


class DrdviRepresentation(_AbstractAutoencoder):
    """Original DRDVI representation model adapted to the ClustPy interface.

    Training uses the stochastic DRDVI variational objective and its row
    orthogonality penalty. Encoding uses the deterministic DRDVI inference path
    and returns the mean of the latent Beta distributions.

    Parameters
    ----------
    layers : list
        Layer sizes from input through hidden layers to the latent dimension,
        for example ``[1024, 256, 128, 64, 32, 16]``.
    orthogonality_weight : float
        Weight of the DRDVI row-orthogonality penalty (default: 1e6).
    variance_weight : float
        Weight of the latent variance anti-collapse penalty (default: 0.0).
    covariance_weight : float
        Weight of the latent off-diagonal covariance penalty (default: 0.0).
    variance_target : float
        Minimum target standard deviation for each latent dimension (default: 0.05).
    work_on_copy : bool
        If True, deep clustering algorithms optimize a copy of this network.
    random_state : np.random.RandomState | int
        Random state used by the inherited ClustPy training routine.
    """

    def __init__(
        self,
        layers: list,
        orthogonality_weight: float = 1e6,
        variance_weight: float = 0.0,
        covariance_weight: float = 0.0,
        variance_target: float = 0.05,
        work_on_copy: bool = True,
        random_state: np.random.RandomState | int = None,
    ):
        super().__init__(work_on_copy=work_on_copy, random_state=random_state)
        if len(layers) < 3:
            raise ValueError("layers must contain input, at least one hidden, and embedding dimensions.")
        if any(not isinstance(dim, int) or dim <= 0 for dim in layers):
            raise ValueError("All entries in layers must be positive integers.")
        if orthogonality_weight < 0:
            raise ValueError("orthogonality_weight must be non-negative.")
        if variance_weight < 0 or covariance_weight < 0:
            raise ValueError("variance_weight and covariance_weight must be non-negative.")
        if variance_target < 0:
            raise ValueError("variance_target must be non-negative.")

        self.layers = layers.copy()
        self.input_dim = self.layers[0]
        self.hidden_dims = self.layers[1:-1]
        self.embedding_dim = self.layers[-1]
        self.n_layers = len(self.hidden_dims)
        self.orthogonality_weight = orthogonality_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target
        self.last_loss_components = {}

        self.sigma = torch.nn.Parameter(torch.tensor([0.0]))
        self.a = torch.nn.Parameter(torch.linspace(-1.0, 1.0, self.n_layers))
        self.A = torch.nn.ParameterList(
            [
                torch.nn.Parameter(self._init_weight(in_dim, out_dim))
                for in_dim, out_dim in zip(self.layers[:-2], self.hidden_dims)
            ]
        )
        self.B = torch.nn.ParameterList(
            [
                torch.nn.Parameter(self._init_weight(out_dim, in_dim))
                for in_dim, out_dim in zip(self.layers[:-2], self.hidden_dims)
            ]
        )
        self.Az = torch.nn.Parameter(self._init_weight(self.hidden_dims[-1], self.embedding_dim))
        self.encoder_a = torch.nn.Linear(self.hidden_dims[-1], self.embedding_dim, bias=False)
        self.encoder_b = torch.nn.Linear(self.hidden_dims[-1], self.embedding_dim, bias=False)

    @staticmethod
    def _init_weight(rows: int, cols: int) -> torch.Tensor:
        bound = math.sqrt(1.0 / cols)
        return torch.empty(rows, cols).uniform_(-bound, bound)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic inference path used by the original DRDVI encoder."""
        h = x.reshape(1, -1) if x.ndim == 1 else x
        if h.shape[1] != self.input_dim:
            raise ValueError(
                f"Input layer of the encoder ({self.input_dim}) does not match input sample ({h.shape[1]})."
            )
        for layer_idx, b_matrix in enumerate(self.B):
            h = torch.sqrt(torch.sigmoid(self.a[layer_idx])) * torch.matmul(h, b_matrix.t())
        return h

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return alpha and beta of the latent Beta distributions."""
        h = self.encode_hidden(x)
        alpha = torch.exp(self.encoder_a(h).clamp(min=-20.0, max=20.0))
        beta = torch.exp(self.encoder_b(h).clamp(min=-20.0, max=20.0))
        return alpha, beta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deterministic DRDVI representation E[z] = alpha / (alpha + beta)."""
        single_sample = x.ndim == 1
        alpha, beta = self.encode_params(x)
        embedded = alpha / (alpha + beta)
        return embedded[0] if single_sample else embedded

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """Return a deterministic reconstruction through the DRDVI generative matrices."""
        single_sample = embedded.ndim == 1
        h = embedded.reshape(1, -1) if single_sample else embedded
        h = torch.matmul(h, self.Az.t())
        for layer_idx, a_matrix in reversed(list(enumerate(self.A))):
            h = torch.matmul(h, a_matrix.t())
            if layer_idx != 0:
                h = torch.relu(h)
        return h[0] if single_sample else h

    def drdvi_loss_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculate the DRDVI objective terms before applying external ENRC weights."""
        batch_size = x.shape[0]
        x0 = x.reshape(batch_size, -1).t()
        a_sig = torch.sigmoid(self.a)
        sigma_squared = torch.exp(self.sigma)

        noise = torch.randn(self.hidden_dims[0], batch_size, device=x.device, dtype=x.dtype)
        a_acc = a_sig[0]
        b_acc = self.B[0]
        xt = torch.sqrt(a_acc) * torch.matmul(b_acc, x0) + torch.sqrt(1.0 - a_acc) * noise
        reconstruction_mean = torch.matmul(self.A[0], xt)
        reconstruction_loss = (x0 - reconstruction_mean).pow(2.0).sum() / batch_size
        reconstruction_loss = reconstruction_loss * 0.5 / sigma_squared + self.input_dim * 0.5 * sigma_squared

        layer_match_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for layer_idx in range(1, self.n_layers):
            previous_a_acc = a_acc
            previous_b_acc = b_acc
            a_acc = a_sig[layer_idx] * a_acc
            b_acc = torch.matmul(self.B[layer_idx], b_acc)

            noise = torch.randn(self.hidden_dims[layer_idx], batch_size, device=x.device, dtype=x.dtype)
            xt = torch.sqrt(a_sig[layer_idx]) * torch.matmul(self.B[layer_idx], xt)
            xt = xt + torch.sqrt(1.0 - a_sig[layer_idx]) * noise
            temp_a = (1.0 - previous_a_acc) / (1.0 - a_acc)

            xhat = torch.relu(torch.matmul(self.A[layer_idx], xt))
            projected_diff = xhat - torch.matmul(previous_b_acc, x0)
            diff = torch.sqrt(previous_a_acc) * projected_diff
            diff = diff - torch.sqrt(previous_a_acc) * a_sig[layer_idx] * temp_a * torch.matmul(
                torch.matmul(self.B[layer_idx].t(), self.B[layer_idx]), projected_diff
            )
            temp = diff.pow(2.0).sum() / (1.0 - previous_a_acc)
            temp = temp + a_sig[layer_idx] / (1.0 - a_sig[layer_idx]) * torch.matmul(
                self.B[layer_idx], diff
            ).pow(2.0).sum()
            layer_match_loss = layer_match_loss + 0.5 * temp / batch_size

        alpha = torch.exp(self.encoder_a(xt.t()).clamp(min=-20.0, max=20.0)).t()
        beta = torch.exp(self.encoder_b(xt.t()).clamp(min=-20.0, max=20.0)).t()
        mean = alpha / (alpha + beta)
        latent_fit = (xt - torch.matmul(self.Az, mean)).pow(2.0).sum()
        covariance_diag = alpha * beta / (alpha + beta).pow(2.0) / (alpha + beta + 1.0)
        covariance = torch.diag(covariance_diag.sum(dim=1))
        covariance_trace = torch.trace(torch.matmul(torch.matmul(self.Az, covariance), self.Az.t()))
        final_layer_loss = (latent_fit + covariance_trace) * 0.5 / (1.0 - a_acc) / batch_size

        prior_loss = (
            (alpha - 1.0) * torch.special.digamma(alpha)
            + (beta - 1.0) * torch.special.digamma(beta)
            - (alpha + beta - 2.0) * torch.special.digamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + torch.lgamma(alpha + beta)
        ).sum() / batch_size

        variational_loss = reconstruction_loss + layer_match_loss + final_layer_loss + prior_loss
        embedded = self.encode(x)
        variance_loss, covariance_loss = self.latent_regularization(embedded)
        return {
            "drdvi_reconstruction": reconstruction_loss.squeeze(),
            "drdvi_layer_match": layer_match_loss.squeeze(),
            "drdvi_final_layer": final_layer_loss.squeeze(),
            "drdvi_prior": prior_loss.squeeze(),
            "drdvi_variational": variational_loss.squeeze(),
            "drdvi_orthogonality": self.orthogonality_penalty().squeeze(),
            "latent_variance": variance_loss.squeeze(),
            "latent_covariance": covariance_loss.squeeze(),
        }

    def _store_loss_components(self, components: dict[str, torch.Tensor]) -> None:
        self.last_loss_components = {
            name: float(value.detach().cpu().item()) for name, value in components.items()
        }

    def drdvi_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the original stochastic DRDVI objective and orthogonality penalty."""
        components = self.drdvi_loss_components(x)
        return components["drdvi_variational"], components["drdvi_orthogonality"]

    def orthogonality_penalty(self) -> torch.Tensor:
        """Measure deviations of the DRDVI inference matrices from row orthogonality."""
        penalty = torch.zeros((), device=self.a.device, dtype=self.a.dtype)
        for b_matrix in self.B:
            eye = torch.eye(b_matrix.shape[0], device=b_matrix.device, dtype=b_matrix.dtype)
            penalty = penalty + (torch.matmul(b_matrix, b_matrix.t()) - eye).pow(2.0).sum()
        return penalty

    def latent_regularization(self, embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate variance and covariance penalties on deterministic embeddings.

        These terms only constrain the representation exposed to clustering.
        They do not alter the stochastic DRDVI diffusion and denoising objective.
        """
        if embedded.ndim != 2:
            raise ValueError("embedded must be a two-dimensional batch x embedding tensor.")

        standard_deviation = torch.sqrt(embedded.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = torch.relu(self.variance_target - standard_deviation).mean()

        centered = embedded - embedded.mean(dim=0)
        denominator = max(embedded.shape[0] - 1, 1)
        covariance = torch.matmul(centered.t(), centered) / denominator
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        covariance_loss = off_diagonal.pow(2.0).sum() / embedded.shape[1]
        return variance_loss, covariance_loss

    def enrc_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full DRDVI regularization term used during joint ENRC training."""
        components = self.drdvi_loss_components(x)
        weighted_orthogonality = self.orthogonality_weight * components["drdvi_orthogonality"] / x.shape[0]
        weighted_variance = self.variance_weight * components["latent_variance"]
        weighted_covariance = self.covariance_weight * components["latent_covariance"]
        total = (
            components["drdvi_variational"]
            + weighted_orthogonality
            + weighted_variance
            + weighted_covariance
        )
        components = components.copy()
        components.update(
            {
                "weighted_orthogonality": weighted_orthogonality,
                "weighted_variance": weighted_variance,
                "weighted_covariance": weighted_covariance,
                "drdvi_enrc_loss": total,
            }
        )
        self._store_loss_components(components)
        return total

    def loss(
        self,
        batch: list,
        ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
        device: torch.device = torch.device("cpu"),
        corruption_fn: Callable = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the full DRDVI objective during ClustPy's network pretraining."""
        del ssl_loss_fn, corruption_fn
        batch_data = batch[1].to(device)
        loss = self.enrc_loss(batch_data)
        embedded = self.encode(batch_data)
        reconstructed = self.decode(embedded)
        return loss, embedded, reconstructed
