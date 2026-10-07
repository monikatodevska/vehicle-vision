"""Class-aware, one-to-one detection evaluation at a fixed IoU threshold."""

from pathlib import Path

import numpy as np

from .annotations import Annotations, read_annotations
from .geometry import pairwise_iou, validate_threshold


def match_detections(predicted: Annotations, actual: Annotations, threshold=0.5):
    validate_threshold(threshold)
    overlaps = pairwise_iou(predicted.boxes, actual.boxes)
    matched = set()
    pairs = []
    for index in np.argsort(-predicted.scores, kind="stable"):
        candidates = [
            target
            for target in range(len(actual.boxes))
            if target not in matched
            and predicted.classes[index] == actual.classes[target]
            and overlaps[index, target] > 0
            and overlaps[index, target] >= threshold
        ]
        if candidates:
            target = max(candidates, key=lambda item: overlaps[index, item])
            matched.add(target)
            pairs.append((int(index), target))
    return pairs


def summarize(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_directories(
    predictions: Path,
    ground_truth: Path,
    *,
    threshold=0.5,
    prediction_format="xyxy",
    ground_truth_format="xywh",
) -> dict:
    validate_threshold(threshold)
    predictions, ground_truth = Path(predictions), Path(ground_truth)
    for directory in (predictions, ground_truth):
        if not directory.is_dir():
            raise ValueError(f"Annotation directory does not exist: {directory}")
    predicted_files = {path.relative_to(predictions) for path in predictions.rglob("*.txt")}
    actual_files = {path.relative_to(ground_truth) for path in ground_truth.rglob("*.txt")}
    names = sorted(predicted_files | actual_files)
    if not names:
        raise ValueError("No .txt annotations found.")
    empty = Annotations(np.empty((0, 4)), np.array([]), np.array([]))
    counts = np.zeros(3, dtype=int)
    per_class = {}
    per_image = {}
    for name in names:
        predicted = (
            read_annotations(predictions / name, prediction_format, predictions=True)
            if name in predicted_files
            else empty
        )
        actual = (
            read_annotations(ground_truth / name, ground_truth_format)
            if name in actual_files
            else empty
        )
        pairs = match_detections(predicted, actual, threshold)
        tp = len(pairs)
        image_counts = (tp, len(predicted.boxes) - tp, len(actual.boxes) - tp)
        counts += image_counts
        per_image[str(name)] = summarize(*image_counts)
        for class_id in np.union1d(predicted.classes, actual.classes):
            class_tp = sum(predicted.classes[index] == class_id for index, _ in pairs)
            class_counts = np.array(
                [
                    class_tp,
                    np.sum(predicted.classes == class_id) - class_tp,
                    np.sum(actual.classes == class_id) - class_tp,
                ],
                dtype=int,
            )
            key = str(int(class_id))
            per_class[key] = per_class.get(key, np.zeros(3, dtype=int)) + class_counts
    return {
        "iou_threshold": threshold,
        "images": len(names),
        "overall": summarize(*(int(value) for value in counts)),
        "per_class": {
            key: summarize(*(int(v) for v in value)) for key, value in sorted(per_class.items())
        },
        "per_image": per_image,
    }
