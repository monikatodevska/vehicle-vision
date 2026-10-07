"""Deterministic, class-aware IoU tracking baseline with one-to-one assignments."""

from dataclasses import dataclass

import numpy as np

from .annotations import Annotations
from .geometry import pairwise_iou, validate_threshold


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    class_id: int
    missed: int = 0


class IoUTracker:
    """Tracks persist for max_missed absent frames; update once per video frame."""

    def __init__(self, threshold=0.3, max_missed=5):
        validate_threshold(threshold)
        if type(max_missed) is not int or max_missed < 0:
            raise ValueError("max_missed must be a non-negative integer.")
        self.threshold = threshold
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1

    def update(self, detections: Annotations) -> list[Track]:
        candidates = []
        boxes = np.asarray([track.box for track in self.tracks]).reshape(-1, 4)
        overlaps = pairwise_iou(boxes, detections.boxes)
        for index, track in enumerate(self.tracks):
            track.missed += 1
            for detection in range(len(detections.boxes)):
                overlap = overlaps[index, detection]
                if (
                    track.class_id == detections.classes[detection]
                    and overlap > 0
                    and overlap >= self.threshold
                ):
                    candidates.append((-overlap, index, detection))
        used_tracks, used_detections = set(), set()
        visible = {}
        for _, index, detection in sorted(candidates):
            if index in used_tracks or detection in used_detections:
                continue
            track = self.tracks[index]
            track.box = detections.boxes[detection].copy()
            track.missed = 0
            used_tracks.add(index)
            used_detections.add(detection)
            visible[detection] = track
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        for index, (box, class_id) in enumerate(zip(detections.boxes, detections.classes)):
            if index not in used_detections:
                track = Track(self.next_id, box.copy(), int(class_id))
                self.next_id += 1
                self.tracks.append(track)
                visible[index] = track
        return [
            Track(visible[index].track_id, visible[index].box.copy(), visible[index].class_id)
            for index in range(len(detections.boxes))
        ]
