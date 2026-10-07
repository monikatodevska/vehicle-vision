"""Bounded-memory directory inference with one model load per run."""

from pathlib import Path

import numpy as np

from .annotations import Annotations, write_annotations
from .config import DetectorConfig
from .detection import decode_predictions


def require_opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install image/video dependencies: pip install -e '.[vision]'"
        ) from error
    return cv2


def predict_directory(model, source: Path, destination: Path, config: DetectorConfig) -> int:
    cv2 = require_opencv()
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f"Image directory does not exist: {source}")
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("Output must be outside the input image directory.")
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not paths:
        raise ValueError("No supported images found.")
    labels = [path.relative_to(source).with_suffix(".txt") for path in paths]
    if len(set(labels)) != len(labels):
        raise ValueError("Images with the same stem would overwrite annotation files.")
    for start in range(0, len(paths), config.batch_size):
        batch_paths = paths[start : start + config.batch_size]
        originals, prepared = [], []
        for path in batch_paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            originals.append(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (config.width, config.height)).astype(np.float32)
            if config.normalize_pixels:
                resized /= 255.0
            prepared.append(resized[..., None])
        outputs = model.predict(np.stack(prepared), verbose=0)
        if isinstance(outputs, dict):
            probabilities, regression = outputs["out_class"], outputs["out_reg"]
        elif isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            output_map = dict(zip(model.output_names, outputs))
            probabilities, regression = output_map["out_class"], output_map["out_reg"]
        else:
            raise ValueError("Model must expose out_class and out_reg outputs.")
        if len(probabilities) != len(batch_paths) or len(regression) != len(batch_paths):
            raise ValueError("Model returned an unexpected batch size.")
        for index, (path, image) in enumerate(zip(batch_paths, originals)):
            detected = decode_predictions(probabilities[index], regression[index], config)
            height, width = image.shape[:2]
            scale = [height / config.height, width / config.width] * 2
            detected = Annotations(detected.boxes * scale, detected.classes, detected.scores)
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            write_annotations(target.with_suffix(".txt"), detected)
            for box, class_id, score in zip(detected.boxes, detected.classes, detected.scores):
                y1, x1, y2, x2 = np.rint(box).astype(int)
                color = ((37 * int(class_id) + 60) % 256, 200, 80)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    f"class {class_id}: {score:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
            if not cv2.imwrite(str(target), image):
                raise ValueError(f"Cannot write image: {target}")
    return len(paths)
