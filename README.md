# DRDVI-ENRC

Research code for combining **Dimension-Reducing Diffusion Variational
Inference (DRDVI)** representation learning with **Deep Embedded
Non-Redundant Clustering (ENRC)**. The repository exposes reproducible
baselines, a frozen two-stage pipeline, and end-to-end joint training.

## Methods

| Name | Representation | Clustering | Updated jointly |
|---|---|---|---|
| `enrc` | ENRC autoencoder | ENRC | yes |
| `drdvi` | DRDVI | KMeans per factor | no |
| `frozen` | pretrained DRDVI | ENRC | no |
| `joint` | pretrained deterministic DRDVI AE | ENRC | encoder + decoder + ENRC |
| `full_joint` | stochastic full-objective DRDVI | ENRC | yes (experimental) |

The primary `joint` method is the validated unfrozen-AE fine-tuning pipeline:
it uses fixed latent standardization and a weighted clustering plus
reconstruction objective. On Stickfigures it improves the difficult third
factor over the decoder-frozen variant. `full_joint` is retained separately as
an experimental implementation combining ENRC with the complete stochastic
DRDVI variational and row-orthogonality objectives.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[test]"
```

Run the synthetic smoke test:

```bash
python examples/quickstart.py
pytest -q
```

## Data

Stickfigures is included under CC BY 4.0 for an immediately runnable example.
Download the remaining ENRC benchmark data from
[Figshare](https://figshare.com/articles/dataset/ENRC/12272921) and follow
[`data/README.md`](data/README.md). Other dataset files and all checkpoints are
excluded from Git by default.

## Experiments

Run every method on every configured dataset:

```bash
python experiments/run_all.py --data-root data --device cpu
```

Use `--epochs 50` to override every pretraining and clustering stage with 50
epochs for a controlled method comparison.

Select datasets or methods:

```bash
python experiments/run_all.py \
  --datasets stickfigures cmnist \
  --methods enrc drdvi frozen joint full_joint \
  --data-root /path/to/enrc_data \
  --device cuda
```

Use `--quick` for a one-epoch pipeline check. It tests wiring, not model
quality. Each job writes `summary_scores.csv`; `run_all.py` combines successful
jobs into `results/comparison/all_summary_scores.csv` and records failed jobs
in `failures.csv`. By default, one failed dataset does not stop the others.

Run only the validated joint method for 500 fine-tuning epochs (it performs a
50-epoch AE-only pretraining stage first):

```bash
python experiments/run_all.py --datasets stickfigures --methods joint \
  --data-root data --epochs 500
```

Run the full stochastic-objective variant separately:

```bash
python experiments/run_all.py --datasets stickfigures --methods full_joint \
  --data-root data --epochs 500
```

With the reference seed and pretrained checkpoint, the validated joint method
reproduces Stickfigures scores of NMI 1.0000, 1.0000, and 0.5259 for
upper-body, lower-body, and the third factor respectively. Checkpoints are not
committed; pass one directly to `experiments/joint_drdvi_enrc.py --checkpoint`
or allow the script to create a new AE-only checkpoint in its output folder.

Dataset-specific parameters live under `configs/`. Edit YAML rather than the
experiment source when changing image size, layer widths, batch size, or epoch
counts.

## Repository layout

```text
src/drdvi_enrc/models/       DRDVI models adapted to the ClustPy interface
src/drdvi_enrc/clustering/   modified ENRC implementation
src/drdvi_enrc/utils/        data, metrics, and reproducibility utilities
experiments/                 baselines, validated joint, full joint, and runner
configs/                     one reproducible configuration per dataset
tests/                       model and integration smoke tests
third_party/                 upstream notices and license texts
```

## Attribution

This repository incorporates and adapts upstream code. See
[`third_party/NOTICE.md`](third_party/NOTICE.md) and the preserved license
files before redistributing modified versions.

- Lukas Miklautz et al., *Deep Embedded Non-Redundant Clustering*, AAAI 2020.
- Junbin Liu, Farzan Farnia, and Wing-Kin Ma, *Multilayer Matrix Factorization
  via Dimension-Reducing Diffusion Variational Inference*, ICML 2025.
- Collin Leiber et al., [ClustPy](https://github.com/collinleiber/ClustPy).

## Status

This is research software. The four-dataset comparison runner is implemented,
but full benchmark values depend on dataset availability, hardware, and final
hyperparameter selection. Report the configuration and random seed with every
result.
