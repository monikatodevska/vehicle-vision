import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vehicle_vision.annotations import Annotations, read_annotations, write_annotations
from vehicle_vision.cli import main
from vehicle_vision.config import DetectorConfig
from vehicle_vision.detection import decode_predictions
from vehicle_vision.evaluation import evaluate_directories, match_detections
from vehicle_vision.geometry import non_max_suppression, pairwise_iou
from vehicle_vision.tracking import IoUTracker


def annotations(boxes, classes=None, scores=None):
    return Annotations(
        np.asarray(boxes).reshape(-1, 4),
        classes if classes is not None else np.zeros(len(boxes)),
        scores if scores is not None else np.ones(len(boxes)),
    )


class GeometryTests(unittest.TestCase):
    def test_overlap_and_empty_boxes(self):
        np.testing.assert_allclose(pairwise_iou([[0, 0, 10, 10]], [[5, 5, 15, 15]]), [[1 / 7]])
        self.assertEqual(pairwise_iou([], [[0, 0, 1, 1]]).shape, (0, 1))
        self.assertEqual(pairwise_iou([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0], 0)

    def test_invalid_coordinates(self):
        for boxes in ([[2, 2, 1, 1]], [[0, 0, np.nan, 1]], [[0, 1]]):
            with self.assertRaises(ValueError):
                pairwise_iou(boxes, [])

    def test_nms_preserves_classes_and_confidence(self):
        boxes = [[0, 0, 10, 10]] * 3
        self.assertEqual(non_max_suppression(boxes, [0.5, 0.9, 0.8], [0, 0, 1]).tolist(), [1, 2])

    def test_nms_stable_ties_and_empty_input(self):
        self.assertEqual(non_max_suppression([[0, 0, 1, 1]] * 2, [0.9, 0.9], [0, 0]).tolist(), [0])
        self.assertEqual(non_max_suppression([], [], []).tolist(), [])


class EvaluationTests(unittest.TestCase):
    def test_class_mismatch_is_not_a_match(self):
        self.assertEqual(
            match_detections(annotations([[0, 0, 5, 5]], [1]), annotations([[0, 0, 5, 5]], [0])), []
        )

    def test_one_to_one_matching_prefers_confidence(self):
        predicted = annotations([[0, 0, 5, 5]] * 2, scores=[0.5, 0.9])
        self.assertEqual(match_detections(predicted, annotations([[0, 0, 5, 5]])), [(1, 0)])

    def test_choose_best_overlap(self):
        predicted = annotations([[0, 0, 10, 10]])
        actual = annotations([[0, 0, 9, 9], [0, 0, 10, 10]])
        self.assertEqual(match_detections(predicted, actual), [(0, 1)])

    def test_zero_threshold_does_not_match_disjoint_boxes(self):
        self.assertEqual(
            match_detections(annotations([[0, 0, 1, 1]]), annotations([[5, 5, 6, 6]]), 0), []
        )

    def test_prediction_only_files_count_as_false_positives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predicted, actual = root / "pred", root / "gt"
            predicted.mkdir()
            actual.mkdir()
            write_annotations(predicted / "extra.txt", annotations([[0, 0, 1, 1]]))
            (actual / "missing.txt").write_text("0,0,1,1,0\n")
            report = evaluate_directories(predicted, actual)
            self.assertEqual(report["overall"]["false_positives"], 1)
            self.assertEqual(report["overall"]["false_negatives"], 1)
            self.assertEqual(report["overall"]["f1"], 0)

    def test_reproducible_example(self):
        root = Path(__file__).resolve().parents[1]
        report = evaluate_directories(root / "examples/predictions", root / "examples/ground_truth")
        self.assertEqual(report["overall"]["true_positives"], 2)
        self.assertEqual(report["overall"]["false_positives"], 2)
        self.assertEqual(report["overall"]["false_negatives"], 1)
        self.assertAlmostEqual(report["overall"]["f1"], 4 / 7)


class AnnotationTests(unittest.TestCase):
    def test_roundtrip_and_size_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boxes.txt"
            original = annotations([[10, 20, 40, 60]], [2], [0.75])
            write_annotations(path, original)
            loaded = read_annotations(path, predictions=True)
            np.testing.assert_allclose(loaded.boxes, original.boxes)
            path.write_text("10,20,30,40,2\n")
            np.testing.assert_allclose(read_annotations(path, "xywh").boxes, original.boxes)

    def test_empty_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boxes.txt"
            path.write_text("\n")
            self.assertEqual(read_annotations(path).boxes.shape, (0, 4))
            for text in ("0,0,1,1,0.5", "0,0,-1,1,0", "0,0,1", "0,0,nan,1,0"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_annotations(path)


class DecoderTests(unittest.TestCase):
    def test_distances_clip_without_mutating_outputs(self):
        config = DetectorConfig(height=8, width=8, background_class=1, regression_mode="distances")
        probabilities = np.array([[[0.95, 0.05]]])
        regression = np.array([[[10.0, 10.0, 10.0, 10.0]]])
        before = regression.copy()
        result = decode_predictions(probabilities, regression, config)
        np.testing.assert_allclose(result.boxes, [[0, 0, 8, 8]])
        np.testing.assert_array_equal(regression, before)

    def test_position_size_and_scaling(self):
        config = DetectorConfig(
            height=8, width=8, background_class=1, regression_scales=(2, 2, 1, 1)
        )
        result = decode_predictions([[[0.95, 0.05]]], [[[1, 0, 0.5, 0.5]]], config)
        np.testing.assert_allclose(result.boxes, [[4, 2, 8, 6]])

    def test_background_and_invalid_boxes_are_removed(self):
        config = DetectorConfig(height=8, width=8, background_class=1)
        self.assertEqual(
            len(decode_predictions([[[0.05, 0.95]]], [[[0, 0, 0.5, 0.5]]], config).boxes), 0
        )
        self.assertEqual(
            len(decode_predictions([[[0.95, 0.05]]], [[[0, 0, -0.5, 0.5]]], config).boxes), 0
        )

    def test_invalid_grid_and_configuration(self):
        with self.assertRaises(ValueError):
            decode_predictions(np.zeros((2, 2, 4)), np.zeros((2, 2, 4)), DetectorConfig())
        for values in (
            {"batch_size": 0},
            {"score_threshold": 2},
            {"stride": 1.5},
            {"regression_scales": [1, 0, 1, 1]},
            {"normalize_pixels": "false"},
        ):
            with self.assertRaises(ValueError):
                DetectorConfig(**values)


class TrackerTests(unittest.TestCase):
    def test_one_to_one_matching_and_class_separation(self):
        tracker = IoUTracker()
        first = tracker.update(annotations([[0, 0, 10, 10]]))
        second = tracker.update(annotations([[0, 0, 10, 10]] * 3, [0, 0, 1]))
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(len({track.track_id for track in second}), 3)

    def test_expiration_and_no_state_aliases(self):
        tracker = IoUTracker(max_missed=1)
        first = tracker.update(annotations([[0, 0, 10, 10]]))
        first[0].box[:] = 100
        tracker.update(annotations([]))
        recovered = tracker.update(annotations([[0, 0, 10, 10]]))
        self.assertEqual(recovered[0].track_id, first[0].track_id)
        tracker.update(annotations([]))
        tracker.update(annotations([]))
        self.assertNotEqual(
            tracker.update(annotations([[0, 0, 10, 10]]))[0].track_id, first[0].track_id
        )


class CLITests(unittest.TestCase):
    def test_evaluation_writes_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / "nested/report.json"
            code = main(
                [
                    "evaluate",
                    "--predictions",
                    str(root / "examples/predictions"),
                    "--ground-truth",
                    str(root / "examples/ground_truth"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())["images"], 2)

    def test_invalid_path_returns_error(self):
        with self.assertLogs(level="ERROR"):
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        "--predictions",
                        "/missing-predictions",
                        "--ground-truth",
                        "/missing-ground-truth",
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
