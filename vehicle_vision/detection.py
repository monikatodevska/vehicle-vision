"""Vectorized decoding of the research detector's two regression conventions."""

import numpy as np

from .annotations import Annotations
from .config import DetectorConfig
from .geometry import non_max_suppression


def decode_predictions(probabilities, regression, config: DetectorConfig) -> Annotations:
    probabilities = np.asarray(probabilities)
    regression = np.asarray(regression)
    if probabilities.ndim != 3 or regression.shape != (*probabilities.shape[:2], 4):
        raise ValueError("Expected HxWxC probabilities and HxWx4 regression.")
    if probabilities.shape[:2] != (config.height // config.stride, config.width // config.stride):
        raise ValueError("Output grid does not match configured image dimensions and stride.")
    if config.background_class >= probabilities.shape[-1]:
        raise ValueError("Background class is outside the model output channels.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Model classification output must contain finite probabilities.")
    if not np.isfinite(regression).all():
        raise ValueError("Model regression output contains non-finite values.")
    classes = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    rows, cols = np.where(
        (confidence >= config.score_threshold) & (classes != config.background_class)
    )
    values = regression[rows, cols].astype(np.float64) * config.regression_scales
    center_y = rows * config.stride + round(config.stride / 2)
    center_x = cols * config.stride + round(config.stride / 2)
    if config.regression_mode == "distances":
        y1, x1 = center_y - values[:, 0], center_x - values[:, 1]
        y2, x2 = center_y + values[:, 2], center_x + values[:, 3]
        valid_values = (values >= 0).all(axis=1)
    else:
        height, width = values[:, 2] * config.height, values[:, 3] * config.width
        y1, x1 = center_y - height / 2 + values[:, 0], center_x - width / 2 + values[:, 1]
        y2, x2 = y1 + height, x1 + width
        valid_values = (height > 0) & (width > 0)
    boxes = np.stack((y1, x1, y2, x2), axis=-1)
    boxes = np.clip(boxes, 0, [config.height, config.width, config.height, config.width])
    valid = valid_values & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, labels = boxes[valid], confidence[rows, cols][valid], classes[rows, cols][valid]
    keep = non_max_suppression(boxes, scores, labels, config.iou_threshold)
    return Annotations(boxes[keep], labels[keep], scores[keep])
