import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vehicle_vision.training import TrainingConfig, validate_training_data
from vehicle_vision.training_data import TrainingSample, load_sample, pair_samples, read_targets


def target_arrays():
    classification = np.zeros((2, 3, 4), dtype=np.float32)
    classification[..., 3] = 1
    classification[0, 0] = [1, 0, 0, 0]
    regression = np.zeros((2, 3, 4), dtype=np.float32)
    regression[0, 0] = [1, 1, 0.25, 0.25]
    return classification, regression


def write_legacy(path, classification, regression):
    with path.open("wb") as stream:
        for tensor in (classification, regression):
            pickle.dump(tensor.shape, stream)
            pickle.dump(tensor.flatten(), stream)


class TargetTests(unittest.TestCase):
    def test_legacy_and_npz_preserve_target_values(self):
        classification, regression = target_arrays()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "image000001.txt"
            write_legacy(legacy, classification, regression)
            with self.assertRaises(ValueError):
                read_targets(legacy, "legacy_pickle")
            actual = read_targets(legacy, "legacy_pickle", trust_pickle=True)
            np.testing.assert_array_equal(actual[0], classification)
            np.testing.assert_array_equal(actual[1], regression)
            path = root / "image000001.npz"
            np.savez(path, classification=classification, regression=regression)
            np.testing.assert_array_equal(read_targets(path, "npz")[0], classification)

    def test_malformed_and_extra_legacy_objects_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.txt"
            path.write_bytes(b"not a pickle")
            with self.assertRaises(ValueError):
                read_targets(path, "legacy_pickle", trust_pickle=True)
            write_legacy(path, *target_arrays())
            with path.open("ab") as stream:
                pickle.dump("extra variant data", stream)
            with self.assertRaises(ValueError):
                read_targets(path, "legacy_pickle", trust_pickle=True)

    def test_pairing_uses_stems_and_rejects_missing_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, targets = root / "images", root / "targets"
            images.mkdir()
            targets.mkdir()
            (images / "image000001.bmp").touch()
            with self.assertRaises(ValueError):
                pair_samples(images, targets, "legacy_pickle")
            (targets / "image000001.txt").touch()
            self.assertEqual(len(pair_samples(images, targets, "legacy_pickle")), 1)
            (images / "image000001.png").touch()
            with self.assertRaises(ValueError):
                pair_samples(images, targets, "legacy_pickle")


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency not installed")
class TrainingDataTests(unittest.TestCase):
    def create_dataset(self, root, target_format="legacy_pickle"):
        import cv2

        paths = {}
        for split, count in (("train", 3), ("validation", 1)):
            images, targets = root / split / "images", root / split / "targets"
            images.mkdir(parents=True)
            targets.mkdir()
            for index in range(count):
                cv2.imwrite(
                    str(images / f"image{index:06d}.bmp"), np.zeros((16, 24), dtype=np.uint8)
                )
                classification, regression = target_arrays()
                if target_format == "legacy_pickle":
                    write_legacy(targets / f"image{index:06d}.txt", classification, regression)
                else:
                    np.savez(
                        targets / f"image{index:06d}.npz",
                        classification=classification,
                        regression=regression,
                    )
            paths[f"{split}_images"] = str(images)
            paths[f"{split}_targets"] = str(targets)
        return TrainingConfig(
            **paths, height=16, width=24, batch_size=2, epochs=1, target_format=target_format
        )

    def test_validation_and_no_silent_resize(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            config = self.create_dataset(Path(temporary))
            training, validation = validate_training_data(config, trust_pickle=True)
            self.assertEqual((len(training), len(validation)), (3, 1))
            cv2.imwrite(str(training[0].image), np.zeros((8, 8), dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_sample(
                    training[0], 16, 24, 4, target_format="legacy_pickle", trust_pickle=True
                )

    def test_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.create_dataset(Path(temporary))
            values = {
                **config.__dict__,
                "validation_images": config.train_images,
                "validation_targets": config.train_targets,
            }
            with self.assertRaises(ValueError):
                validate_training_data(TrainingConfig(**values), trust_pickle=True)

    def test_invalid_target_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.create_dataset(Path(temporary))
            image = Path(config.train_images) / "image000000.bmp"
            target = Path(config.train_targets) / "image000000.txt"
            classification, regression = target_arrays()
            write_legacy(target, classification[:, :, :1], regression)
            with self.assertRaises(ValueError):
                load_sample(
                    TrainingSample(image, target),
                    16,
                    24,
                    4,
                    target_format="legacy_pickle",
                    trust_pickle=True,
                )


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow") and importlib.util.find_spec("cv2"),
    "TensorFlow and OpenCV optional dependencies not installed",
)
class TrainingModelTests(TrainingDataTests):
    def test_partial_batch_is_kept(self):
        import tensorflow as tf

        from vehicle_vision.training import build_batches

        with tempfile.TemporaryDirectory() as temporary:
            config = self.create_dataset(Path(temporary))
            samples, _ = validate_training_data(config, trust_pickle=True)
            batches = build_batches(samples, config, tf, shuffle=False, trust_pickle=True)
            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[1][0].shape[0], 1)
            np.testing.assert_array_equal(batches[0][1]["out_reg"][0], target_arrays()[1])

    def test_losses_match_original_equations_and_ignore_zero_targets(self):
        from vehicle_vision.training_losses import LegacyFocalLoss, LegacyRegressionLoss

        actual = np.array([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]], dtype=np.float32)
        predicted = np.full_like(actual, 0.25)
        expected = -np.log(0.25) * (1 - 0.25) ** 2 / 2
        self.assertAlmostEqual(float(LegacyFocalLoss()(actual, predicted)), expected, places=6)
        targets = np.array([[[[0.0, 1.0, 2.0, 0.0]]]], dtype=np.float32)
        np.testing.assert_allclose(LegacyRegressionLoss()(targets, np.zeros_like(targets)), 0.02)
        self.assertEqual(float(LegacyRegressionLoss()(np.zeros_like(targets), targets)), 0)

    def test_one_epoch_training_and_saved_model_reload(self):
        from vehicle_vision.models import load_detector
        from vehicle_vision.training import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.create_dataset(root)
            report = train(config, root / "result", trust_pickle=True)
            self.assertEqual(report["epochs_completed"], 1)
            restored = load_detector(root / "result/best.keras")
            self.assertEqual(restored.output_names, ["out_class", "out_reg"])
            self.assertTrue((root / "result/history.csv").is_file())
            metadata = json.loads((root / "result/run.json").read_text())
            self.assertEqual(metadata["training_samples"], 3)
            with self.assertRaises(ValueError):
                train(config, root / "result", trust_pickle=True)
