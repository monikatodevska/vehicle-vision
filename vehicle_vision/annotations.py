"""Explicit, validated CSV annotation formats, without pickle dependencies."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import boxes_array


@dataclass(frozen=True)
class Annotations:
    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        boxes = boxes_array(self.boxes)
        classes = np.asarray(self.classes, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        if classes.shape != (len(boxes),) or scores.shape != classes.shape:
            raise ValueError("Each box needs one class and score.")
        if not np.isfinite(classes).all() or np.any(classes < 0) or np.any(classes % 1):
            raise ValueError("Classes must be non-negative integers.")
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Scores must be probabilities between 0 and 1.")
        object.__setattr__(self, "boxes", boxes.copy())
        object.__setattr__(self, "classes", classes.astype(np.int64))
        object.__setattr__(self, "scores", scores.copy())


def read_annotations(path: Path, box_format="xyxy", *, predictions=False) -> Annotations:
    """Read y,x,ymax,xmax,class[,score] or y,x,height,width,class[,score]."""
    if box_format not in {"xyxy", "xywh"}:
        raise ValueError("Box format must be xyxy or xywh (row/y first).")
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            expected = {5, 6} if predictions else {5}
            if len(row) not in expected:
                raise ValueError(f"{path}:{number}: expected {sorted(expected)} columns.")
            try:
                values = [float(cell) for cell in row]
            except ValueError as error:
                raise ValueError(f"{path}:{number}: expected numeric values.") from error
            rows.append(values if len(values) == 6 else values + [1.0])
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    boxes = data[:, :4].copy()
    if box_format == "xywh":
        boxes[:, 2:] += boxes[:, :2]
    return Annotations(boxes, data[:, 4], data[:, 5])


def write_annotations(path: Path, annotations: Annotations) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for box, class_id, score in zip(annotations.boxes, annotations.classes, annotations.scores):
            writer.writerow([*(f"{coord:.4f}" for coord in box), int(class_id), f"{score:.6f}"])
