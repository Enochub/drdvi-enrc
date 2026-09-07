from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from msc_latent import MultipleNonRedundantSpectralClustering


FACTOR_NAMES = {
    "cmnist": ["left_digit", "right_digit"],
    "nr_objects": ["color", "material", "shape"],
    "stickfigures": ["upper_body", "lower_body", "third_label"],
}


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def pairwise_scores(truth: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nmi = np.asarray([[normalized_mutual_info_score(truth[:, r], predictions[:, q])
                       for r in range(truth.shape[1])]
                      for q in range(predictions.shape[1])])
    ari = np.asarray([[adjusted_rand_score(truth[:, r], predictions[:, q])
                       for r in range(truth.shape[1])]
                      for q in range(predictions.shape[1])])
    return nmi, ari


def save_matrix(path: Path, matrix: np.ndarray, row_names: list[str], column_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["learned_view", *column_names])
        for name, row in zip(row_names, matrix):
            writer.writerow([name, *map(float, row)])


def main(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latent_raw = np.load(args.latent).astype(np.float64)
    labels = np.load(args.labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    if len(latent_raw) != len(labels):
        raise ValueError("latent and labels have different sample counts")
    factor_names = FACTOR_NAMES[args.dataset]
    if labels.shape[1] != len(factor_names):
        raise ValueError("label columns do not match dataset metadata")

    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    init_seed = args.seed if args.init_seed is None else args.init_seed
    kmeans_seed = args.seed if args.kmeans_seed is None else args.kmeans_seed

    # Fixed random sampling is label-free; labels are sliced only after indices exist.
    rng = np.random.RandomState(sample_seed)
    count = min(args.sample_size, len(latent_raw))
    indices = np.sort(rng.choice(len(latent_raw), size=count, replace=False))
    latent = StandardScaler().fit_transform(latent_raw[indices])
    truth = labels[indices]
    clusters = args.n_clusters or [int(np.unique(truth[:, i]).size) for i in range(truth.shape[1])]
    if len(clusters) != len(args.subspace_dims):
        raise ValueError("n-clusters and subspace-dims must have equal lengths")

    model = MultipleNonRedundantSpectralClustering(
        n_clusters=clusters,
        subspace_dims=args.subspace_dims,
        lambda_hsic=args.lambda_hsic,
        sigma=args.sigma if args.sigma == "median" else float(args.sigma),
        max_iter=args.max_iter,
        tol=args.tol,
        learning_rate=args.learning_rate,
        stiefel_update=args.stiefel_update,
        random_state=init_seed,
        kmeans_random_state=kmeans_seed,
        device=args.device,
        verbose=True,
    ).fit(latent)

    nmi, ari = pairwise_scores(truth, model.labels_)
    rows, columns = linear_sum_assignment(-nmi)
    matching = [{
        "learned_view": int(q), "ground_truth_view": int(r),
        "ground_truth_name": factor_names[r], "nmi": float(nmi[q, r]),
        "ari": float(ari[q, r]),
    } for q, r in zip(rows, columns)]
    orthogonality = [float(np.linalg.norm(w.T @ w - np.eye(w.shape[1]))) for w in model.W_]

    overlap_rows = []
    for q in range(len(model.W_)):
        for r in range(q + 1, len(model.W_)):
            overlap_rows.append({
                "view_q": q, "view_r": r,
                "frobenius_overlap": float(np.linalg.norm(model.W_[q].T @ model.W_[r]) ** 2),
                "principal_angles_radians": [float(x) for x in subspace_angles(model.W_[q], model.W_[r])],
            })

    with (args.output_dir / "subspace_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("view_q", "view_r", "frobenius_overlap", "principal_angles_radians"),
        )
        writer.writeheader()
        writer.writerows({
            **row,
            "principal_angles_radians": json.dumps(row["principal_angles_radians"]),
        } for row in overlap_rows)

    row_names = [f"view_{q}" for q in range(len(model.W_))]
    save_matrix(args.output_dir / "pairwise_nmi.csv", nmi, row_names, factor_names)
    save_matrix(args.output_dir / "pairwise_ari.csv", ari, row_names, factor_names)
    np.save(args.output_dir / "sample_indices.npy", indices)
    for q, (w, u) in enumerate(zip(model.W_, model.U_)):
        np.save(args.output_dir / f"W_view_{q}.npy", w)
        np.save(args.output_dir / f"P_view_{q}.npy", w @ w.T)
        np.save(args.output_dir / f"U_view_{q}.npy", u)
        np.save(args.output_dir / f"predictions_view_{q}.npy", model.labels_[:, q])

    with (args.output_dir / "objective_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model.objective_history_[0]))
        writer.writeheader()
        writer.writerows(model.objective_history_)

    # Diagnostics/baselines are evaluated pairwise but do not enter mSC fitting.
    diagnostic = {}
    predictions_by_k = {}
    spectral_by_k = {}
    for k in sorted(set(clusters)):
        predictions_by_k[k] = KMeans(n_clusters=k, n_init=20, random_state=kmeans_seed).fit_predict(latent)
        spectral_by_k[k] = SpectralClustering(
            n_clusters=k, affinity="rbf", gamma=1.0, assign_labels="kmeans", random_state=kmeans_seed
        ).fit_predict(latent)
    diagnostic["kmeans_raw_nmi"] = [float(normalized_mutual_info_score(truth[:, r], predictions_by_k[clusters[r]])) for r in range(len(clusters))]
    diagnostic["spectral_raw_nmi"] = [float(normalized_mutual_info_score(truth[:, r], spectral_by_k[clusters[r]])) for r in range(len(clusters))]
    diagnostic["kmeans_projected_nmi"] = [
        [float(normalized_mutual_info_score(truth[:, r], KMeans(n_clusters=clusters[q], n_init=20, random_state=kmeans_seed).fit_predict(latent @ model.W_[q])))
         for r in range(truth.shape[1])] for q in range(len(model.W_))
    ]

    result = {
        "latent_shape": list(latent.shape), "n_views": len(clusters),
        "sample_seed": sample_seed, "init_seed": init_seed,
        "kmeans_seed": kmeans_seed,
        "n_clusters": clusters, "subspace_dims": args.subspace_dims,
        "sigmas": model.sigmas_, "stiefel_update": args.stiefel_update,
        "orthogonality_errors": orthogonality,
        "matching": matching,
        "mean_matched_nmi": float(np.mean([item["nmi"] for item in matching])),
        "mean_matched_ari": float(np.mean([item["ari"] for item in matching])),
        "subspace_overlap": overlap_rows, "diagnostics": diagnostic,
    }
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matching[0]))
        writer.writeheader()
        writer.writerows(matching)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({
        **vars(args), "latent": str(args.latent), "labels": str(args.labels),
        "output_dir": str(args.output_dir),
    }, indent=2), encoding="utf-8")
    print(f"MSC_RESULT {json.dumps(result)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc mSC on a frozen latent representation")
    parser.add_argument("--dataset", choices=tuple(FACTOR_NAMES), required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--n-clusters", type=parse_int_list)
    parser.add_argument("--subspace-dims", type=parse_int_list, required=True)
    parser.add_argument("--lambda-hsic", type=float, default=1.0)
    parser.add_argument("--sigma", default="median")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--stiefel-update", choices=("qr", "matrix_exp"), default="qr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--init-seed", type=int)
    parser.add_argument("--kmeans-seed", type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
