"""Keras model construction derived from archive/research/helper_model1.py.

The inception detector retains the original layer ordering and output names.
"""

from pathlib import Path


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from error
    return tf


def build_detector(input_shape=(341, 512, 1), num_classes=4):
    """Build the stride-8 inception detector, including one background class."""
    if len(input_shape) != 3 or any(type(dim) is not int or dim <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain three positive integer dimensions.")
    if min(input_shape[:2]) < 8 or type(num_classes) is not int or num_classes < 2:
        raise ValueError("Need at least 8x8 input and two classes including background.")
    tf = require_tensorflow()
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape)

    def convolution(tensor, filters, kernel=3, initializer="normal"):
        return layers.Conv2D(
            filters, kernel, padding="same", activation="relu", kernel_initializer=initializer
        )(tensor)

    features = inputs
    for filters in (16, 48, 48):
        features = convolution(features, filters)
        features = convolution(features, filters)
        features = layers.MaxPool2D(2)(features)
    features = convolution(features, 48, initializer="he_normal")
    branch_3 = convolution(features, 48, initializer="he_normal")
    branch_1 = convolution(features, 48, 1, "he_normal")
    branch_5 = convolution(features, 48, 5, "he_normal")
    features = layers.Concatenate()([branch_1, branch_3, branch_5])
    classifier = convolution(features, 64)
    classifier = layers.Conv2D(
        num_classes, 1, activation="softmax", kernel_initializer="uniform", name="out_class"
    )(classifier)
    regressor = convolution(features, 64)
    regressor = layers.Conv2D(
        4, 1, activation="linear", kernel_initializer="zeros", name="out_reg"
    )(regressor)
    return tf.keras.Model(inputs, [classifier, regressor], name="vehicle_detector")


def load_detector(path: Path, weights: Path | None = None):
    """Load a .keras model, or a legacy JSON architecture plus separate weights."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Model does not exist: {path}")
    if weights is not None and not Path(weights).is_file():
        raise ValueError(f"Weights do not exist: {weights}")
    tf = require_tensorflow()
    if path.suffix == ".json":
        if weights is None:
            raise ValueError("A JSON architecture requires --weights.")
        model = tf.keras.models.model_from_json(path.read_text(encoding="utf-8"))
        model.load_weights(str(weights))
        return model
    if weights is not None:
        raise ValueError("Use --weights only with a JSON architecture.")
    return tf.keras.models.load_model(str(path), compile=False)
