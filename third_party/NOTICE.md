# Third-party notices

This repository combines and adapts ideas and code from three upstream works.

- **ENRC** — Deep Embedded Non-Redundant Clustering, Lukas Miklautz et al.
  The original implementation is MIT licensed. See `ENRC_LICENSE`.
- **DRDVI** — Multilayer Matrix Factorization via Dimension-Reducing Diffusion
  Variational Inference, Junbin Liu, Farzan Farnia and Wing-Kin Ma. The
  original implementation is MIT licensed. See `DRDVI_LICENSE`.
- **ClustPy** — the ENRC implementation and its interfaces are derived from
  ClustPy, copyright Collin Leiber and contributors, under BSD-3-Clause. See
  `CLUSTPY_LICENSE`.

The files under `src/drdvi_enrc/models/` adapt DRDVI to the ClustPy neural
network interface. `src/drdvi_enrc/clustering/enrc.py` is a modified ClustPy
ENRC module. Modifications add loss-history support and compatibility with the
DRDVI representation objective. Upstream copyright notices remain applicable.

