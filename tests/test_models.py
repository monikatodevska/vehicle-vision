import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vehicle_vision.models import build_detector, load_detector


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow"), "TensorFlow optional dependency not installed"
)
class ModelTests(unittest.TestCase):
    def test_model_outputs_and_serialization(self):
        model = build_detector((16, 24, 1), num_classes=4)
        inputs = np.zeros((1, 16, 24, 1), dtype=np.float32)
        outputs = model(inputs, training=False)
        self.assertEqual(model.output_names, ["out_class", "out_reg"])
        self.assertEqual(tuple(outputs[0].shape), (1, 2, 3, 4))
        np.testing.assert_allclose(np.asarray(outputs[0]).sum(axis=-1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.keras"
            model.save(path)
            restored = load_detector(path)
            np.testing.assert_allclose(restored(inputs)[0], outputs[0], atol=1e-6)
