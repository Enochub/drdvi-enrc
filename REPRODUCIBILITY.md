# Reproducibility guide

This file distinguishes portable entry points in the repository from the
long-running jobs used to produce the thesis tables.

## Scope and labels

The primary datasets are C-MNIST, NR-Objects, and Stickfigures. Their evaluated
cluster-count configurations are `10/10`, `6/2/3`, and `3/3`, respectively.
The third label present in the Stickfigures source file is not part of the
primary evaluation.

Labels are not used to train DR-DVI, ENRC, NrKMeans, or Nr-DipMeans. They are
used for ACC, NMI, ARI, view matching, linear probing, and visual colouring.
Linear probing and UMAP are therefore diagnostic procedures rather than
unsupervised training stages.

## Software environments

- Neural experiments: Python, PyTorch, scikit-learn, and the adapted ENRC and
  DR-DVI modules under `src/`.
- NrKMeans: ClustPy.
- Official Nr-DipMeans: Java 11, Scala 2.13, SBT 1.4.6, and the upstream Scala
  project. The upstream project is not vendored here; provide its path with
  `--scala-project`.
- GPU jobs were run through Slurm. Fixed-representation methods run on CPU where
  applicable.

## Inputs excluded from Git

The following files must be generated or downloaded locally:

- raw C-MNIST and NR-Objects images and labels;
- contiguous image caches used to reduce network-filesystem overhead;
- pretrained DR-DVI checkpoints;
- latent matrices and sample-level predictions.

The fixed-representation scripts expect `latent.npy` with shape `(n, d_z)` and
`labels.npy` with shape `(n, M)`. Rows must have the same sample order.

## Experiment-to-script map

| Thesis experiment | Entry point | Notes |
|---|---|---|
| Standard ENRC | `experiments/frozen_drdvi_enrc.py --methods enrc` | seed 42 reference runs |
| DR-DVI pretraining | `experiments/frozen_drdvi_enrc.py --methods drdvi` | exports/uses a fixed latent |
| Frozen DR-DVI + ENRC | `experiments/frozen_drdvi_enrc.py --methods frozen` | encoder not updated |
| Joint DR-DVI + ENRC | `experiments/joint_drdvi_enrc.py` | C-MNIST repeated with seeds 42, 43, 44 |
| NrKMeans | `experiments/fixed_nrkmeans.py` | standardized fixed latent |
| Nr-DipMeans | `experiments/fixed_nrdipmeans.py` | official Scala implementation |
| Linear probe | `experiments/linear_probe.py` | supervised diagnostic only |
| UMAP | `experiments/latent_umap.py` | nonlinear visualization only |
| Prototype-distance bridge | `experiments/prototype_distance_bridge.py` | fixed NR-Objects representation; not standard joint training |
| mSC | `experiments/run_msc_latent.py` | preliminary 300-sample C-MNIST diagnostic |

## Seeds and aggregation

Unless stated otherwise, tables report seed 42. The main joint C-MNIST result
uses seeds 42, 43, and 44. Dataset-level values are arithmetic means over the
evaluated views; the three-seed result uses the arithmetic mean and sample
standard deviation over runs.

Nr-DipMeans uses ten internal runs. The selected run minimizes the
unsupervised NrKMeans cost, not a ground-truth metric.

## Matching

Cluster identifiers are matched one-to-one for ACC. Predicted spaces are
matched one-to-one to reference factors. NR-Objects primary evaluation enforces
equal cluster counts for methods whose counts are specified. Nr-DipMeans is
matched without this restriction because it estimates the counts; every
estimated count must therefore be reported with its metrics.

## Expected summaries

The small CSV files in `results/summary/` record the values transcribed into the
thesis. They are provided for auditability and do not replace raw outputs.
Floating-point results may vary with hardware and dependency versions.
