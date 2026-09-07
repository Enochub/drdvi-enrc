"""Load the three benchmark datasets used in the thesis experiments."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.preprocessing import LabelEncoder

DATASET_NAMES = ("stickfigures", "cmnist", "nr_objects")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _subset(paths: list[Path], maximum: int | None, seed: int) -> list[Path]:
    if maximum is None or len(paths) <= maximum:
        return paths
    indices = np.sort(np.random.default_rng(seed).choice(len(paths), maximum, replace=False))
    return [paths[index] for index in indices]


def _image(path: Path, size: int, grayscale: bool) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L" if grayscale else "RGB").resize((size, size), Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 255.0).reshape(-1)


def _encoded(columns: list[tuple[str, list]]) -> tuple[np.ndarray, list[str]]:
    return np.column_stack([LabelEncoder().fit_transform(values) for _, values in columns]), [
        name for name, _ in columns
    ]


def load_dataset(name: str, data_root: str | Path, config: dict, seed: int = 42):
    root = Path(data_root)
    maximum = config.get("max_samples")
    size = int(config.get("image_size", 64))
    grayscale = bool(config.get("grayscale", False))
    if name == "stickfigures":
        data = np.loadtxt(root / name / "stickfigures_3sub.data", delimiter=";").astype(np.float32)
        if maximum and len(data) > maximum:
            indices = np.sort(np.random.default_rng(seed).choice(len(data), maximum, replace=False))
            data = data[indices]
        # The source file has a third label. The primary thesis evaluation
        # follows the two-view ENRC benchmark and excludes it.
        return data[:, 3:] / 255.0, data[:, :2].astype(np.int64), ["upper_body", "lower_body"]
    if name == "cmnist":
        paths = _subset(sorted((root / name).rglob("*.png")), maximum, seed)
        labels = [tuple(map(int, path.parent.name.split("_"))) for path in paths]
        return np.stack([_image(p, size, grayscale) for p in paths]), np.asarray(labels), ["left_digit", "right_digit"]
    if name == "nr_objects":
        base = root / name
        paths = _subset(sorted((base / "images" / "train").glob("*.png")), maximum, seed)
        objects = [json.loads((base / "scenes" / f"{p.stem}.json").read_text(encoding="utf-8"))["objects"][0] for p in paths]
        labels, names = _encoded([(key, [obj[key] for obj in objects]) for key in ("color", "material", "shape")])
        return np.stack([_image(p, size, grayscale) for p in paths]), labels, names
    raise ValueError(f"Unsupported dataset: {name}")
