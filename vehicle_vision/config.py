"""Validated inference settings loaded from portable JSON files."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import validate_threshold


@dataclass(frozen=True)
class DetectorConfig:
    height: int = 341
    width: int = 512
    stride: int = 8
    score_threshold: float = 0.85
    iou_threshold: float = 0.3
    background_class: int = 3
    regression_mode: str = "position_size"
    regression_scales: tuple = (1.0, 1.0, 1.0, 1.0)
    normalize_pixels: bool = False
    batch_size: int = 8

    def __post_init__(self):
        for name in ("height", "width", "stride", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.height < self.stride or self.width < self.stride:
            raise ValueError("Image dimensions must be at least the output stride.")
        if type(self.background_class) is not int or self.background_class < 0:
            raise ValueError("background_class must be a non-negative integer.")
        if type(self.normalize_pixels) is not bool:
            raise ValueError("normalize_pixels must be true or false.")
        validate_threshold(self.score_threshold)
        validate_threshold(self.iou_threshold)
        if self.regression_mode not in {"position_size", "distances"}:
            raise ValueError("regression_mode must be position_size or distances.")
        scales = np.asarray(self.regression_scales, dtype=float)
        if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("regression_scales must contain four finite positive values.")
        object.__setattr__(self, "regression_scales", tuple(scales.tolist()))

    @classmethod
    def load(cls, path: Path):
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object.")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid configuration fields: {error}") from error
