"""Bounding box operations using (ymin, xmin, ymax, xmax) pixel coordinates."""

import numpy as np


def boxes_array(boxes) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        if array.shape not in {(0,), (0, 4)}:
            raise ValueError("Boxes must have shape (N, 4).")
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("Boxes must have shape (N, 4).")
    if not np.isfinite(array).all():
        raise ValueError("Box coordinates must be finite.")
    if np.any(array[:, 2:] < array[:, :2]):
        raise ValueError("Box maximum coordinates must not precede minimum coordinates.")
    return array


def pairwise_iou(boxes_a, boxes_b) -> np.ndarray:
    """Compute all pair overlaps; zero-area boxes have zero IoU."""
    a, b = boxes_array(boxes_a), boxes_array(boxes_b)
    starts = np.maximum(a[:, None, :2], b[None, :, :2])
    ends = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(ends - starts, 0).prod(axis=-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def validate_threshold(value: float) -> None:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Threshold must be between 0 and 1.")


def non_max_suppression(boxes, scores, classes, iou_threshold=0.3) -> np.ndarray:
    """Keep highest-confidence boxes independently per class, with stable ties."""
    validate_threshold(iou_threshold)
    boxes = boxes_array(boxes)
    scores, classes = np.asarray(scores), np.asarray(classes)
    if scores.shape != (len(boxes),) or classes.shape != scores.shape:
        raise ValueError("Each box must have one score and one class.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities between 0 and 1.")
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        remaining = order[1:]
        overlaps = pairwise_iou(boxes[current : current + 1], boxes[remaining])[0]
        order = remaining[(classes[remaining] != classes[current]) | (overlaps <= iou_threshold)]
    return np.asarray(keep, dtype=np.int64)
