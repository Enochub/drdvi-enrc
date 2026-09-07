from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.cluster import KMeans


class MultipleNonRedundantSpectralClustering:
    """Exact-kernel post-hoc mSC for a fixed two-dimensional representation.

    This follows Niu, Dy, and Jordan (ICML 2010), except that the paper's
    dimensionally problematic right matrix-exponential update is replaced by
    either a QR retraction or a left-action skew-symmetric matrix exponential.
    """

    def __init__(
        self,
        n_clusters: list[int] | tuple[int, ...],
        subspace_dims: list[int] | tuple[int, ...],
        lambda_hsic: float = 1.0,
        sigma: float | str = "median",
        max_iter: int = 10,
        tol: float = 1e-4,
        learning_rate: float = 0.1,
        stiefel_update: str = "qr",
        armijo_c: float = 1e-4,
        backtracking_factor: float = 0.5,
        max_backtracking: int = 12,
        random_state: int = 42,
        kmeans_random_state: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> None:
        self.n_clusters = list(n_clusters)
        self.subspace_dims = list(subspace_dims)
        self.lambda_hsic = float(lambda_hsic)
        self.sigma = sigma
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = float(learning_rate)
        self.stiefel_update = stiefel_update
        self.armijo_c = float(armijo_c)
        self.backtracking_factor = float(backtracking_factor)
        self.max_backtracking = int(max_backtracking)
        self.random_state = int(random_state)
        self.kmeans_random_state = (
            self.random_state if kmeans_random_state is None else int(kmeans_random_state)
        )
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = bool(verbose)

    @staticmethod
    def _rbf_kernel(projected: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        squared_norm = (projected * projected).sum(dim=1, keepdim=True)
        squared_distance = squared_norm + squared_norm.T - 2 * projected @ projected.T
        squared_distance = squared_distance.clamp_min(0)
        kernel = torch.exp(-squared_distance / (2 * sigma * sigma))
        return (kernel + kernel.T) * 0.5

    @staticmethod
    def _normalized_similarity(kernel: torch.Tensor) -> torch.Tensor:
        degree = kernel.sum(dim=1).clamp_min(torch.finfo(kernel.dtype).eps)
        inv_sqrt_degree = degree.rsqrt()
        normalized = inv_sqrt_degree[:, None] * kernel * inv_sqrt_degree[None, :]
        return (normalized + normalized.T) * 0.5

    @staticmethod
    def _center_kernel(kernel: torch.Tensor) -> torch.Tensor:
        row_mean = kernel.mean(dim=1, keepdim=True)
        return kernel - row_mean - row_mean.T + kernel.mean()

    @classmethod
    def _hsic(cls, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        denominator = max(first.shape[0] - 1, 1) ** 2
        return (cls._center_kernel(first) * cls._center_kernel(second)).sum() / denominator

    def _resolve_sigmas(self, z: torch.Tensor, matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(self.sigma, str):
            if self.sigma != "median":
                raise ValueError("sigma must be positive or 'median'")
            sigmas = []
            with torch.no_grad():
                for matrix in matrices:
                    projected = z @ matrix
                    distances = torch.pdist(projected)
                    positive = distances[distances > 0]
                    value = positive.median() if positive.numel() else torch.ones((), device=z.device, dtype=z.dtype)
                    sigmas.append(value.clamp_min(torch.finfo(z.dtype).eps))
            return sigmas
        value = float(self.sigma)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("sigma must be positive or 'median'")
        return [torch.tensor(value, device=z.device, dtype=z.dtype) for _ in matrices]

    def _kernels(self, z: torch.Tensor, matrices: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self._rbf_kernel(z @ matrix, sigma) for matrix, sigma in zip(matrices, self.sigmas_)]

    def _spectral_embeddings(
        self,
        kernels: list[torch.Tensor],
        row_normalize: bool = False,
    ) -> list[torch.Tensor]:
        embeddings = []
        with torch.no_grad():
            for kernel, clusters in zip(kernels, self.n_clusters):
                normalized = self._normalized_similarity(kernel)
                _, vectors = torch.linalg.eigh(normalized)
                embedding = vectors[:, -clusters:]
                if row_normalize:
                    embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(
                        torch.finfo(embedding.dtype).eps
                    )
                embeddings.append(embedding)
        return embeddings

    def _objective(
        self,
        z: torch.Tensor,
        matrices: list[torch.Tensor],
        embeddings: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels = self._kernels(z, matrices)
        spectral = sum(
            torch.trace(embedding.T @ self._normalized_similarity(kernel) @ embedding)
            for embedding, kernel in zip(embeddings, kernels)
        )
        # The paper sums q != r. Count each unordered pair twice accordingly.
        hsic = 2 * sum(
            (self._hsic(kernels[q], kernels[r]) for q in range(len(kernels)) for r in range(q + 1, len(kernels))),
            start=torch.zeros((), device=z.device, dtype=z.dtype),
        )
        return spectral - self.lambda_hsic * hsic, spectral, hsic

    def _retract(self, matrix: torch.Tensor, gradient: torch.Tensor, step: float) -> torch.Tensor:
        if self.stiefel_update == "qr":
            symmetric = (matrix.T @ gradient + gradient.T @ matrix) * 0.5
            tangent = gradient - matrix @ symmetric
            candidate, triangular = torch.linalg.qr(matrix + step * tangent, mode="reduced")
            signs = torch.sign(torch.diagonal(triangular)).clamp(min=-1, max=1)
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return candidate * signs
        if self.stiefel_update == "matrix_exp":
            skew = gradient @ matrix.T - matrix @ gradient.T
            return torch.matrix_exp(step * skew) @ matrix
        raise ValueError("stiefel_update must be 'qr' or 'matrix_exp'")

    def _update_one(
        self,
        z: torch.Tensor,
        matrices: list[torch.Tensor],
        embeddings: list[torch.Tensor],
        view: int,
    ) -> list[torch.Tensor]:
        variable = matrices[view].detach().clone().requires_grad_(True)
        trial_matrices = [item.detach() for item in matrices]
        trial_matrices[view] = variable
        objective, _, _ = self._objective(z, trial_matrices, embeddings)
        gradient = torch.autograd.grad(objective, variable)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite gradient in view {view}")
        with torch.no_grad():
            old_value = float(objective)
            symmetric = (variable.T @ gradient + gradient.T @ variable) * 0.5
            tangent = gradient - variable @ symmetric
            directional = float((gradient * tangent).sum().clamp_min(0))
            step = self.learning_rate
            accepted = None
            for _ in range(self.max_backtracking + 1):
                candidate = self._retract(variable, gradient, step)
                candidates = [item.detach() for item in matrices]
                candidates[view] = candidate
                candidate_value = float(self._objective(z, candidates, embeddings)[0])
                if math.isfinite(candidate_value) and candidate_value >= old_value + self.armijo_c * step * directional:
                    accepted = candidate
                    break
                step *= self.backtracking_factor
            if accepted is None:
                accepted = matrices[view].detach()
        result = [item.detach() for item in matrices]
        result[view] = accepted.detach()
        return result

    def fit(self, z: np.ndarray | torch.Tensor) -> "MultipleNonRedundantSpectralClustering":
        z_tensor = torch.as_tensor(z, dtype=self.dtype, device=self.device).detach()
        if z_tensor.ndim != 2 or z_tensor.shape[0] < 2 or not torch.isfinite(z_tensor).all():
            raise ValueError("z must be a finite two-dimensional array with at least two rows")
        if len(self.n_clusters) != len(self.subspace_dims) or not self.n_clusters:
            raise ValueError("n_clusters and subspace_dims must have the same non-zero length")
        if any(c < 2 or c >= z_tensor.shape[0] for c in self.n_clusters):
            raise ValueError("each cluster count must be in [2, n_samples)")
        if any(dim < 1 or dim > z_tensor.shape[1] for dim in self.subspace_dims):
            raise ValueError("each subspace dimension must be in [1, n_features]")

        generator = torch.Generator(device=self.device).manual_seed(self.random_state)
        matrices = []
        for dimension in self.subspace_dims:
            random = torch.randn(z_tensor.shape[1], dimension, generator=generator, device=self.device, dtype=self.dtype)
            matrices.append(torch.linalg.qr(random, mode="reduced")[0])
        self.sigmas_ = self._resolve_sigmas(z_tensor, matrices)
        self.objective_history_ = []

        previous = None
        for iteration in range(self.max_iter):
            # The objective constraint is U.T @ U = I. Row normalization is the
            # Ng-Jordan-Weiss post-processing step and is used only for KMeans;
            # feeding it back into the W objective makes the spectral term scale
            # with n and breaks the alternating optimization interpretation.
            embeddings = self._spectral_embeddings(
                self._kernels(z_tensor, matrices), row_normalize=False
            )
            for view in range(len(matrices)):
                matrices = self._update_one(z_tensor, matrices, embeddings, view)
            objective, spectral, hsic = self._objective(z_tensor, matrices, embeddings)
            row = {
                "iteration": iteration,
                "total_objective": float(objective.detach()),
                "spectral_quality": float(spectral.detach()),
                "hsic_penalty": float(hsic.detach()),
            }
            if not all(math.isfinite(value) for key, value in row.items() if key != "iteration"):
                raise FloatingPointError("Non-finite objective")
            self.objective_history_.append(row)
            if self.verbose:
                print(f"MSC_ITERATION {row}", flush=True)
            current = row["total_objective"]
            if previous is not None and abs(current - previous) <= self.tol * max(1.0, abs(previous)):
                break
            previous = current

        final_kernels = self._kernels(z_tensor, matrices)
        embeddings = self._spectral_embeddings(final_kernels, row_normalize=True)
        self.W_ = [matrix.cpu().numpy() for matrix in matrices]
        self.U_ = [embedding.cpu().numpy() for embedding in embeddings]
        self.labels_ = np.column_stack([
            KMeans(
                n_clusters=clusters,
                n_init=20,
                random_state=self.kmeans_random_state,
            ).fit_predict(embedding)
            for clusters, embedding in zip(self.n_clusters, self.U_)
        ]).astype(np.int32)
        self.sigmas_ = [float(value.cpu()) for value in self.sigmas_]
        self.n_iter_ = len(self.objective_history_)
        self.n_features_in_ = z_tensor.shape[1]
        return self

    def fit_predict(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.fit(z).labels_
