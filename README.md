# DR-DVI representations for non-redundant clustering

Code and reproducibility material for the master thesis **Multiple
Clusterings with Diffusion in Latent Space**. The experiments study whether
representations learned with Dimension-Reducing Diffusion Variational
Inference (DR-DVI) support multiple non-redundant clusterings and whether
they are compatible with Embedded Non-Redundant Clustering (ENRC).

The repository accompanies the thesis; it is research code rather than a
general-purpose library. Raw datasets and neural-network checkpoints are not
included.

## Experimental scope

The thesis uses three datasets and the following reference views:

| Dataset | Samples | Evaluated views | Clusters per view |
|---|---:|---|---:|
| C-MNIST | 60,000 | left digit, right digit | 10 / 10 |
| NR-Objects | 10,000 | color, material, shape | 6 / 2 / 3 |
| Stickfigures | 900 | upper body, lower body | 3 / 3 |

The Stickfigures source file contains a third label. It is retained in the
data file but deliberately excluded from the primary thesis evaluation to
match the two-view ENRC benchmark protocol.

The evaluated pipelines are:

| Pipeline | Representation | Encoder during clustering | Purpose |
|---|---|---|---|
| Standard ENRC | original input | trainable ENRC autoencoder | reference baseline |
| Fixed DR-DVI clustering | standardized pretrained latent | frozen | KMeans, NrKMeans, and Nr-DipMeans diagnostics |
| Frozen DR-DVI + ENRC | pretrained DR-DVI latent | frozen | two-stage integration |
| Joint DR-DVI + ENRC | DR-DVI representation | trainable | direct joint integration |

The prototype-distance bridge and multiple spectral clustering (mSC) are
exploratory diagnostics. They are not presented as established baselines.

## Evaluation protocol

Every predicted partition is evaluated with clustering accuracy (ACC),
normalized mutual information (NMI), and adjusted Rand index (ARI). Cluster
identifiers are aligned by one-to-one Hungarian matching when ACC is
calculated.

Predicted clustering spaces have no semantic order. C-MNIST and Stickfigures
views are matched one-to-one by maximum NMI. For NR-Objects, primary matching
respects the distinct reference cardinalities 6, 2, and 3. Nr-DipMeans is the
exception because it estimates its own cluster counts; its spaces are matched
without a cardinality restriction and the estimated count is always reported
beside the metrics. Ground-truth labels are used only for post-hoc evaluation,
unless a script explicitly identifies a diagnostic analysis.

Most thesis experiments use seed 42. The reported joint C-MNIST experiment
uses seeds 42, 43, and 44 and reports the mean and sample standard deviation.
Nr-DipMeans performs ten internal runs and selects the run with the lowest
unsupervised NrKMeans cost.

## Repository layout

```text
configs/                    dataset-level defaults
data/                       download and preparation instructions
experiments/                training and evaluation entry points
results/summary/            small, thesis-facing result summaries
src/drdvi_enrc/             adapted DR-DVI and ENRC components
tests/                      smoke and metric tests
third_party/                upstream attribution and licence notices
```

The main entry points are:

```text
experiments/frozen_drdvi_enrc.py      standard ENRC and frozen pipelines
experiments/joint_drdvi_enrc.py       joint deterministic DR-DVI--ENRC
experiments/joint_full_drdvi_enrc.py  experimental full DR-DVI objective
experiments/fixed_nrkmeans.py         NrKMeans on a saved latent matrix
experiments/fixed_nrdipmeans.py       official Scala Nr-DipMeans wrapper
experiments/linear_probe.py           post-hoc information-retention probe
experiments/latent_umap.py            post-hoc UMAP visualization
experiments/prototype_distance_bridge.py  exploratory bridge construction
experiments/run_msc_latent.py          exploratory multiple spectral clustering
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the mapping between thesis
experiments, commands, external software, and expected input files.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,visualization]"
pytest -q
```

CUDA is optional for smoke tests and fixed-representation clustering, but GPU
execution is recommended for neural representation learning.

## Data

Download C-MNIST, NR-Objects, and Stickfigures as described in
[`data/README.md`](data/README.md). Dataset caches, saved latent matrices, and
checkpoints are excluded from Git. The loaders do not use ground-truth labels
during representation learning or clustering.

## Running the compact pipelines

Run a one-epoch wiring check on Stickfigures:

```bash
python experiments/run_all.py \
  --datasets stickfigures \
  --methods enrc drdvi frozen joint \
  --data-root data \
  --device cpu \
  --quick
```

Run selected pipelines with the dataset configuration files:

```bash
python experiments/run_all.py \
  --datasets cmnist nr_objects stickfigures \
  --methods enrc drdvi frozen joint \
  --data-root /path/to/prepared/data \
  --device cuda \
  --seed 42
```

These commands provide portable experiment entry points. Exact long-running
settings used for thesis tables, including the Slurm environment and external
Nr-DipMeans implementation, are documented in `REPRODUCIBILITY.md`.

## Fixed-representation clustering

NrKMeans accepts a latent matrix and labels in NumPy format:

```bash
python experiments/fixed_nrkmeans.py \
  --latent path/to/latent.npy \
  --labels path/to/labels.npy \
  --factor-names color material shape \
  --clusters 6 2 3 \
  --output-dir results/nr_objects/nrkmeans \
  --seed 42
```

Nr-DipMeans additionally requires Java 11, Scala 2.13, SBT 1.4.6, and a local
checkout of the official implementation:

```bash
python experiments/fixed_nrdipmeans.py \
  --scala-project /path/to/NrKmeans_and_NrDipmeans \
  --latent path/to/latent.npy \
  --labels path/to/labels.npy \
  --factor-names color material shape \
  --output-dir results/nr_objects/nrdipmeans \
  --seed 42
```

## Diagnostics

Linear probing and UMAP are post-hoc analyses. Their use of labels must not be
confused with unsupervised model fitting.

```bash
python experiments/linear_probe.py \
  --latent path/to/latent.npy --labels path/to/labels.npy \
  --factor-names color material shape --output results/linear_probe.json

python experiments/latent_umap.py \
  --latent path/to/latent.npy --labels path/to/labels.npy \
  --factor-names color material shape --output figures/nr_objects_umap.png
```

Construct the exploratory NR-Objects prototype-distance representation:

```bash
python experiments/prototype_distance_bridge.py \
  --latent path/to/latent.npy --labels path/to/labels.npy \
  --factor-names color material shape --clusters 6 2 3 \
  --output-dir results/nr_objects/prototype_bridge --seed 42
```

The mSC runner exposes the preliminary C-MNIST experiment. It uses a fixed,
label-free sample before labels are accessed for evaluation:

```bash
python experiments/run_msc_latent.py \
  --dataset cmnist --latent path/to/latent.npy --labels path/to/labels.npy \
  --output-dir results/cmnist/msc --n-clusters 10,10 \
  --subspace-dims 4,4 --sample-size 300 --sigma 2.5 \
  --lambda-hsic 1800 --max-iter 50 --seed 42
```

## Reproducing reported numbers

The repository does not promise bit-for-bit equality across hardware and
library versions. Use the recorded seeds, preprocessing, cluster counts, and
matching protocol, and report the generated configuration with every result.
Small summaries of the values used in the thesis are kept in
[`results/summary`](results/summary); raw predictions and checkpoints are not.

The principal dataset-level values are summarized below. These values are
included to make the connection between the code repository and thesis tables
explicit.

| Dataset | Standard ENRC NMI | Fixed NrKMeans NMI | Fixed Nr-DipMeans NMI |
|---|---:|---:|---:|
| C-MNIST | 0.6998 | 0.5298 | 0.4440 |
| NR-Objects | 0.8644 | 0.3772 | 0.5027 |
| Stickfigures | 1.0000 | 1.0000 | 0.6084 |

Nr-DipMeans estimated `9/5`, `2/2/2`, and `17/3` clusters on C-MNIST,
NR-Objects, and Stickfigures. Its NMI must therefore be interpreted together
with the estimated cluster counts, ACC, and ARI. In particular, the larger
NR-Objects NMI does not mean that the intended `6/2/3` structure was recovered.

Direct DR-DVI--ENRC integration was dataset dependent. The joint C-MNIST mean
was ACC/NMI/ARI `0.4168/0.3543/0.2406` across seeds 42, 43, and 44. Frozen and
joint NR-Objects means were `0.3919/0.0384/0.0197` and
`0.3793/0.0105/0.0132`, while both evaluated Stickfigures configurations
reached `1.0000/1.0000/1.0000`.

## Attribution

The repository incorporates and adapts research code from ENRC, DR-DVI, and
ClustPy. See [`third_party/NOTICE.md`](third_party/NOTICE.md) and the preserved
licence files before redistributing modified sources.

- Lukas Miklautz et al., *Deep Embedded Non-Redundant Clustering*, AAAI 2020.
- Junbin Liu, Farzan Farnia, and Wing-Kin Ma, *Multilayer Matrix Factorization
  via Dimension-Reducing Diffusion Variational Inference*, ICML 2025.
- Collin Leiber et al., [ClustPy](https://github.com/collinleiber/ClustPy).

## Citation

If this repository is used, cite the thesis and the original DR-DVI and ENRC
publications. Repository metadata is provided in [`CITATION.cff`](CITATION.cff).

## Status

This repository is an archival research release. The conclusions in the thesis
are limited to the reported checkpoints, seeds, and evaluation protocols.
