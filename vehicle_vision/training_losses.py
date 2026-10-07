"""Serializable equivalents of the original two-head research losses."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="vehicle_vision")
class LegacyFocalLoss(tf.keras.losses.Loss):
    """Original categorical focal loss with unit alpha and gamma=2 by default."""

    def __init__(self, gamma=2.0, name="legacy_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        probabilities = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        return tf.reduce_mean(
            tf.reduce_sum(
                -y_true * tf.pow(1 - probabilities, self.gamma) * tf.math.log(probabilities),
                axis=-1,
            )
        )

    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma}


@tf.keras.utils.register_keras_serializable(package="vehicle_vision")
class LegacyRegressionLoss(tf.keras.losses.Loss):
    """Preserve rpn_loss_reg: zero-target masking, summed smooth L1, weight 0.01."""

    def __init__(self, weight=0.01, name="legacy_regression_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        difference = (y_true - y_pred) * tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        absolute = tf.abs(difference)
        penalties = tf.where(absolute <= 1, 0.5 * tf.square(difference), absolute - 0.5)
        return self.weight * tf.reduce_sum(penalties)

    def get_config(self):
        return {**super().get_config(), "weight": self.weight}
