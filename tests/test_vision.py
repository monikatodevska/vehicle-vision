import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vehicle_vision.annotations import read_annotations
from vehicle_vision.config import DetectorConfig
from vehicle_vision.inference import predict_directory
from vehicle_vision.video import extract_frames


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency not installed")
class VisionTests(unittest.TestCase):
    def test_inference_batches_and_scales_to_original_dimensions(self):
        import cv2

        class FakeModel:
            output_names = ["out_reg", "out_class"]

            def __init__(self):
                self.batch_sizes = []

            def predict(self, batch, verbose=0):
                self.batch_sizes.append(len(batch))
                if batch.dtype != np.float32:
                    raise AssertionError("Inference inputs must use float32")
                return [
                    np.tile([[[[2.0, 2.0, 2.0, 2.0]]]], (len(batch), 1, 1, 1)),
                    np.tile([[[[0.95, 0.05]]]], (len(batch), 1, 1, 1)),
                ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "images"
            source.mkdir()
            for index in range(3):
                cv2.imwrite(str(source / f"{index}.png"), np.zeros((16, 24, 3), dtype=np.uint8))
            config = DetectorConfig(
                height=8, width=8, background_class=1, regression_mode="distances", batch_size=2
            )
            model = FakeModel()
            self.assertEqual(predict_directory(model, source, root / "result", config), 3)
            self.assertEqual(model.batch_sizes, [2, 1])
            result = read_annotations(root / "result/0.txt", predictions=True)
            np.testing.assert_allclose(result.boxes, [[4, 6, 12, 18]])

    def test_streaming_video_sampling_and_manifest(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "video.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
            if not writer.isOpened():
                self.skipTest("MJPG codec unavailable")
            try:
                for _ in range(5):
                    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            self.assertEqual(extract_frames(path, root / "frames", every=2), 3)
            self.assertIn("frame_00000004.jpg,4,0.400000", (root / "frames/frames.csv").read_text())
            with self.assertRaises(ValueError):
                extract_frames(path, root / "frames", every=2)
