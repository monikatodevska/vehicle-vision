# Vehicle Vision

Vehicle detection and traffic analysis tools built from a collection of fully
convolutional single-shot detector research experiments. The maintained Python
package provides image inference, detection evaluation, video frame extraction,
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
- Automated tests and lint/format checks configured for GitHub Actions.

## Repository Layout

```text
vehicle_vision/       Maintained application and reusable components
configs/             Portable inference configuration
examples/            Synthetic annotations for a reproducible evaluation
tests/               Core, image/video, and optional model tests
docs/                Architecture, migration notes, and research inventory
archive/research/    Historical experiments retained locally (excluded from Git)
.github/workflows/   Automated checks
```

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
job installs TensorFlow and checks output shapes and save/load round trips.
Archive scripts are excluded from the supported package, lint checks, and test
discovery because they retain experiment-specific dependencies and paths.

See [architecture](docs/architecture.md), [migration notes](docs/migration.md),
and [GitHub publishing instructions](docs/publishing.md).

## Project Status

This project develops vehicle detection and traffic-analysis experiments from
an initial machine-vision baseline. The published package consolidates reusable
operations and provides a reproducible evaluation workflow. Historical training
experiments are preserved locally and excluded from the public repository.

No model accuracy, throughput, or real-world speed-estimation claims are made
without validated data and measurements. A license has not yet been selected.
