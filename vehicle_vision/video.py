"""Streaming frame extraction without loading a whole video into memory."""

import math
from pathlib import Path

from .inference import require_opencv


def extract_frames(video: Path, destination: Path, every=1) -> int:
    if type(every) is not int or every <= 0:
        raise ValueError("Frame interval must be a positive integer.")
    video, destination = Path(video), Path(destination)
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError("Frame output directory must be empty to avoid mixing videos.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        valid_fps = math.isfinite(fps) and fps > 0
        with (destination / "frames.csv").open("w", encoding="utf-8") as manifest:
            manifest.write("filename,source_frame,timestamp_seconds\n")
            index, saved = 0, 0
            while True:
                success, image = capture.read()
                if not success:
                    break
                if index % every == 0:
                    name = f"frame_{index:08d}.jpg"
                    if not cv2.imwrite(str(destination / name), image):
                        raise ValueError(f"Cannot write frame: {name}")
                    timestamp = f"{index / fps:.6f}" if valid_fps else ""
                    manifest.write(f"{name},{index},{timestamp}\n")
                    saved += 1
                index += 1
        if not saved:
            raise ValueError("Video contains no readable frames.")
        return saved
    finally:
        capture.release()
