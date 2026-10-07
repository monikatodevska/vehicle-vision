"""Portable training of the original two-head, stride-8 inception detector."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .models import build_detector, load_detector, require_tensorflow
from .training_data import load_sample, pair_samples


@dataclass(frozen=True)
class TrainingConfig:
    train_images: str
    train_targets: str
    validation_images: str
    validation_targets: str
    height: int = 341
    width: int = 512
    num_classes: int = 4
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 0.0001
    seed: int = 5
    target_format: str = "legacy_pickle"
    normalize_pixels: bool = False
    focal_gamma: float = 2.0
    regression_weight: float = 0.01
    initial_model: str | None = None
    initial_weights: str | None = None

    def __post_init__(self):
        for field in ("height", "width", "num_classes", "batch_size", "epochs"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer.")
        if min(self.height, self.width) < 8 or self.num_classes < 2:
            raise ValueError("Need at least 8x8 input and two classes including background.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        for field in ("learning_rate", "focal_gamma", "regression_weight"):
            value = getattr(self, field)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive.")
        if self.target_format not in {"legacy_pickle", "npz"}:
            raise ValueError("target_format must be legacy_pickle or npz.")
        if type(self.normalize_pixels) is not bool:
            raise ValueError("normalize_pixels must be a boolean.")
        for field in ("train_images", "train_targets", "validation_images", "validation_targets"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty path.")
        for field in ("initial_model", "initial_weights"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be null or a non-empty path.")
        if self.initial_weights and not self.initial_model:
            raise ValueError("initial_weights requires initial_model.")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Training configuration must be a JSON object.")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid training fields: {error}") from error


def sample_options(config, trust_pickle):
    return dict(
        height=config.height,
        width=config.width,
        num_classes=config.num_classes,
        target_format=config.target_format,
        normalize_pixels=config.normalize_pixels,
        trust_pickle=trust_pickle,
    )


def validate_training_data(config: TrainingConfig, *, trust_pickle=False):
    training = pair_samples(config.train_images, config.train_targets, config.target_format)
    validation = pair_samples(
        config.validation_images, config.validation_targets, config.target_format
    )
    training_paths = {sample.image.resolve() for sample in training}
    if training_paths & {sample.image.resolve() for sample in validation}:
        raise ValueError("Training and validation splits share image paths.")
    target_paths = {sample.target.resolve() for sample in training}
    if target_paths & {sample.target.resolve() for sample in validation}:
        raise ValueError("Training and validation splits share target paths.")
    for sample in training + validation:
        load_sample(sample, **sample_options(config, trust_pickle))
    return training, validation


def build_batches(samples, config, tf, *, shuffle, trust_pickle=False):
    class TensorBatches(tf.keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(config.seed)
            self.indices = np.arange(len(samples))
            if shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self):
            return (len(samples) + config.batch_size - 1) // config.batch_size

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            selected = self.indices[index * config.batch_size : (index + 1) * config.batch_size]
            batch = [
                load_sample(samples[item], **sample_options(config, trust_pickle))
                for item in selected
            ]
            return (
                np.stack([image for image, _ in batch]),
                {
                    name: np.stack([targets[name] for _, targets in batch])
                    for name in ("out_class", "out_reg")
                },
            )

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.indices)

    return TensorBatches()


def train(config: TrainingConfig, destination: Path, *, trust_pickle=False):
    """Validate targets, train bounded batches, and save reproducible run artifacts."""
    training, validation = validate_training_data(config, trust_pickle=trust_pickle)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Training output must be empty to preserve previous runs.")
    tf = require_tensorflow()
    from .training_losses import LegacyFocalLoss, LegacyRegressionLoss

    tf.keras.utils.set_random_seed(config.seed)
    model = (
        load_detector(
            Path(config.initial_model),
            Path(config.initial_weights) if config.initial_weights else None,
        )
        if config.initial_model
        else build_detector((config.height, config.width, 1), config.num_classes)
    )
    if tuple(model.input_shape[1:]) != (config.height, config.width, 1):
        raise ValueError("Initial model input shape does not match training configuration.")
    output_shapes = dict(zip(model.output_names, model.output_shape))
    grid = (config.height // 8, config.width // 8)
    expected = {"out_class": (*grid, config.num_classes), "out_reg": (*grid, 4)}
    if (
        any(
            name not in output_shapes or tuple(output_shapes[name][1:]) != shape
            for name, shape in expected.items()
        )
        or len(output_shapes) != 2
    ):
        raise ValueError("Initial model must have matching out_class and out_reg heads.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "out_class": LegacyFocalLoss(config.focal_gamma),
            "out_reg": LegacyRegressionLoss(config.regression_weight),
        },
    )
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "tensorflow_version": tf.__version__,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "training_ids": [str(sample.image) for sample in training],
        "validation_ids": [str(sample.image) for sample in validation],
    }
    (destination / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(destination / "best.keras"), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(str(destination / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        build_batches(training, config, tf, shuffle=True, trust_pickle=trust_pickle),
        validation_data=build_batches(
            validation, config, tf, shuffle=False, trust_pickle=trust_pickle
        ),
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if not all(np.isfinite(value) for values in history.history.values() for value in values):
        raise ValueError("Training produced non-finite metrics; inspect the recorded run.")
    model.save(destination / "final.keras")
    return {
        "training_samples": len(training),
        "validation_samples": len(validation),
        "epochs_completed": len(history.epoch),
        "best_model": str(destination / "best.keras"),
        "final_model": str(destination / "final.keras"),
    }
