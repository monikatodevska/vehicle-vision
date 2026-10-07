# Vehicle Vision

Vehicle detection and traffic analysis tools built from a collection of fully
convolutional single-shot detector research experiments. The maintained Python
package provides model training, image inference, detection evaluation, video frame extraction,
and an IoU tracking baseline.

**Author:** Monika Todevska.

**Stack:** Python, NumPy, OpenCV, optional TensorFlow/Keras, GitHub Actions.

## Try It

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vision,dev]'
vehicle-vision evaluate \
  --predictions examples/predictions \
  --ground-truth examples/ground_truth \
  --output outputs/example-report.json
```

On Windows, activate with `.venv\Scripts\activate`.
Without installing the entry point, use `python -m vehicle_vision`.

The included **synthetic** example has two correct detections, two false
positives, and one false negative. Expected precision is 0.5, recall is 0.6667,
and F1 is 0.5714. These numbers demonstrate the evaluator; they are not model
benchmark results. The example needs neither TensorFlow nor trained weights.

## Features

- Vectorized bounding box decoding and IoU calculations.
- Confidence-ranked non-maximum suppression applied independently per class.
- Class-aware, one-to-one evaluation with overall, per-class, and per-image reports.
- Image inference in configurable batches using float32 inputs.
- Video extraction with a source-frame and timestamp manifest.
- Deterministic IoU tracking with persistent IDs and configurable track expiration.
- An inception-based, stride-8 detector adapted from the original research architecture.
- Portable two-head training with validated legacy targets, checkpoints, and run logs.
- Original training pipelines, generated-ground-truth loaders, custom losses, and model variants.
- Automated tests and lint/format checks configured for GitHub Actions.

## Repository Layout

```text
vehicle_vision/       Maintained application and reusable components
configs/             Portable inference and training configurations
examples/            Synthetic annotations for a reproducible evaluation
tests/               Core, image/video, and optional model tests
docs/                Architecture, migration notes, and research inventory
archive/research/    Full research source, including training and ground-truth generation
.github/workflows/   Automated checks
```

## Training And Ground Truth

The professional two-head trainer is in [vehicle_vision/training.py](vehicle_vision/training.py).
It preserves the original model and loss conventions while adding data validation,
bounded batches, reproducible settings, and saved checkpoints. After installing
`.[ml]` and configuring your private image and tensor folders:

```bash
vehicle-vision validate-training --config configs/training.example.json --trust-pickle
vehicle-vision train --config configs/training.example.json \
  --output outputs/training/run-01 --trust-pickle
```

Use the trust flag only for your own generated pickle files. Safe NPZ targets
are also supported. See the [training guide](docs/training.md) for target formats,
configuration, fine-tuning, and artifacts. All original experiment code remains
in [archive/research](archive/research):

| Pipeline | Source |
| --- | --- |
| Separate classification and regression heads | [FCN_SSD_anchorless_generator.py](archive/research/FCN_SSD_anchorless_generator.py) |
| Combined classification and regression output | [FCN_SSD_anchorless_generator_combined_output.py](archive/research/FCN_SSD_anchorless_generator_combined_output.py) |
| Classification masks with weight maps | [FCN_SSD_generator_new_LS.py](archive/research/FCN_SSD_generator_new_LS.py) |
| Model architectures and custom losses | [helper_model1.py](archive/research/helper_model1.py), [helper_losses.py](archive/research/helper_losses.py) |

These research scripts retain their experiment-specific paths, preprocessing,
and historical Keras APIs. Their generated `.txt` ground-truth files can be
binary pickle streams; they are different from the bounding-box CSV files used
by the maintained evaluator. Training needs the corresponding private images,
generated tensors, configuration, and a compatible environment.

## Run Inference

Install the optional machine-learning dependencies:

```bash
python -m pip install -e '.[ml]'
vehicle-vision predict \
  --model models/detector.keras \
  --config configs/detector.example.json \
  --images data/test-images \
  --output outputs/detections
```

Legacy JSON architectures can be loaded with `--model models/model.json
--weights models/model.h5`. The model must expose the two outputs `out_class`
and `out_reg`. The configuration must match its input dimensions, output stride,
background class, regression convention, normalization coefficients, and pixel
preprocessing. The example configuration is a starting point, not a verified
configuration for an included checkpoint.

Results contain annotated images and six-column prediction files in original
image coordinates. Subdirectory names are retained. Each model is loaded once,
and only one image batch is prepared at a time.

Trained weights, research datasets, and footage are excluded from Git. The
original `cp.hdf5` checkpoint remains locally in the archive; its provenance and
compatibility have not been verified. Older custom layers, wavelet inputs,
combined-output models, and training pipelines remain in the research archive.

## Video And Evaluation

```bash
vehicle-vision extract-frames \
  --video data/traffic.mp4 --output outputs/frames --every 5

vehicle-vision evaluate \
  --predictions outputs/detections \
  --ground-truth data/annotations \
  --ground-truth-format xywh \
  --prediction-format xyxy \
  --iou 0.5 --output outputs/report.json
```

Annotations use headerless comma-separated text files. **Coordinates always
start with row/y, then column/x**, following the original research convention:

| Format | Columns |
| --- | --- |
| `xywh` | `ymin,xmin,height,width,class_id` |
| `xyxy` | `ymin,xmin,ymax,xmax,class_id` |
| Prediction | Either coordinate format, followed by `class_id,confidence` |

The format names describe corner versus size encoding; they do not change the
y-first coordinate order. Confidence is optional for existing prediction files
and defaults to 1.0. Empty files represent no objects. The evaluator pairs files
by their relative paths, includes prediction-only files as false positives, and
treats missing prediction files as no detections. It reports precision, recall,
and F1 at one IoU threshold, not COCO mAP.

## Development

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

TensorFlow tests skip when the ML dependency is absent. The GitHub Actions model
job installs TensorFlow and checks output shapes, synthetic training, loss
equations, and save/load round trips.
Archive scripts are excluded from the supported package, lint checks, and test
discovery because they retain experiment-specific dependencies and paths.

See [architecture](docs/architecture.md), [migration notes](docs/migration.md),
and [GitHub publishing instructions](docs/publishing.md).

## Project Status

This project develops vehicle detection and traffic-analysis experiments from
an initial machine-vision baseline. The published package consolidates reusable
operations and provides a reproducible evaluation workflow. Historical training
experiments, supporting utilities, and custom losses are included in the public
research archive. Private data and model binaries remain local.

No model accuracy, throughput, or real-world speed-estimation claims are made
without validated data and measurements. A license has not yet been selected.
