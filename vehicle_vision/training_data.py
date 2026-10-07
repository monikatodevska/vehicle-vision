"""Load precomputed research tensors without regenerating ground truth."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .inference import require_opencv


@dataclass(frozen=True)
class TrainingSample:
    image: Path
    target: Path


def pair_samples(images: Path, targets: Path, target_format: str) -> list[TrainingSample]:
    images, targets = Path(images), Path(targets)
    if target_format not in {"legacy_pickle", "npz"}:
        raise ValueError("target_format must be legacy_pickle or npz.")
    if not images.is_dir() or not targets.is_dir():
        raise ValueError("Image and ground-truth directories must exist.")
    suffix = ".txt" if target_format == "legacy_pickle" else ".npz"
    image_files = {}
    for image in sorted(images.rglob("*")):
        if image.is_file() and image.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}:
            key = image.relative_to(images).with_suffix("")
            if key in image_files:
                raise ValueError(f"Duplicate image stem: {key}")
            image_files[key] = image
    target_files = {
        path.relative_to(targets).with_suffix(""): path
        for path in targets.rglob(f"*{suffix}")
        if path.is_file()
    }
    missing = image_files.keys() - target_files.keys()
    orphaned = target_files.keys() - image_files.keys()
    if missing or orphaned:
        raise ValueError(
            f"Unpaired data: {len(missing)} images without targets, "
            f"{len(orphaned)} targets without images."
        )
    if not image_files:
        raise ValueError("No matching training samples found.")
    return [TrainingSample(image_files[key], target_files[key]) for key in sorted(image_files)]


def read_targets(path: Path, target_format: str, *, trust_pickle=False):
    path = Path(path)
    if target_format == "legacy_pickle":
        if not trust_pickle:
            raise ValueError("Legacy targets require --trust-pickle for your own generated files.")
        try:
            with path.open("rb") as stream:
                class_shape = pickle.load(stream)
                classification = np.asarray(pickle.load(stream), dtype=np.float32).reshape(
                    class_shape
                )
                regression_shape = pickle.load(stream)
                regression = np.asarray(pickle.load(stream), dtype=np.float32).reshape(
                    regression_shape
                )
                if stream.read(1):
                    raise ValueError("Extra target objects: select the matching research variant.")
        except (EOFError, pickle.UnpicklingError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid two-head ground truth in {path}: {error}") from error
    elif target_format == "npz":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"classification", "regression"}:
                raise ValueError("NPZ targets require classification and regression arrays.")
            classification = np.asarray(data["classification"], dtype=np.float32)
            regression = np.asarray(data["regression"], dtype=np.float32)
    else:
        raise ValueError("Unsupported target format.")
    return classification, regression


def load_sample(
    sample: TrainingSample,
    height,
    width,
    num_classes,
    *,
    target_format,
    normalize_pixels=False,
    trust_pickle=False,
):
    cv2 = require_opencv()
    image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (height, width):
        raise ValueError(
            f"{sample.image}: image must be readable and exactly {height}x{width}; "
            "resizing would misalign precomputed targets."
        )
    classification, regression = read_targets(
        sample.target, target_format, trust_pickle=trust_pickle
    )
    grid = (height // 8, width // 8)
    if classification.shape != (*grid, num_classes) or regression.shape != (*grid, 4):
        raise ValueError(
            f"{sample.target}: expected classification {(*grid, num_classes)} "
            f"and regression {(*grid, 4)}."
        )
    if not np.isfinite(classification).all() or not np.isfinite(regression).all():
        raise ValueError(f"{sample.target}: targets must be finite.")
    sums = classification.sum(axis=-1)
    if np.any((classification < 0) | (classification > 1)) or not np.all(
        np.isclose(sums, 0) | np.isclose(sums, 1)
    ):
        raise ValueError(f"{sample.target}: classification cells must sum to 1 or 0 (ignored).")
    image = image.astype(np.float32)[..., None]
    if normalize_pixels:
        image /= 255.0
    return image, {"out_class": classification, "out_reg": regression}
