# Data

The small Stickfigures benchmark is included so the primary example can run
immediately. It originates from the ENRC dataset published by Lukas Miklautz
and collaborators on Figshare:

- Source: https://figshare.com/articles/dataset/ENRC/12272921
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Included file: `stickfigures/stickfigures_3sub.data`
- SHA-256: `AC290DA5798324CDAA91E2306F4024B02A51872055FE0194FFE518EBFDB54097`

Please cite the ENRC dataset and paper when using or redistributing this file.
The other benchmark datasets are intentionally not committed. Download and
prepare them as described by the ENRC benchmark material, then arrange them as
follows:

```text
data/
  cmnist/**/*.png
  nr_objects/images/train/*.png
  nr_objects/scenes/*.json
  stickfigures/stickfigures_3sub.data  # already included
```

Use `--data-root` to point the experiment scripts to a different directory.
The C-MNIST loader expects the two integer digit labels in the image's parent
directory name, separated by an underscore. Each NR-Objects scene JSON must
contain one object with `color`, `material`, and `shape` fields.

Although `stickfigures_3sub.data` contains three label columns, only the first
two (upper body and lower body) are used by the primary thesis protocol.

The repository's MIT license applies to original code; the included
Stickfigures data remains separately licensed under CC BY 4.0.
