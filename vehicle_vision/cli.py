"""Command-line interface with optional dependencies loaded on demand."""

import argparse
import json
import logging
from pathlib import Path

from . import __version__
from .evaluation import evaluate_directories


def create_parser():
    parser = argparse.ArgumentParser(description="Vehicle detection and evaluation tools")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Compare CSV predictions with ground truth")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument("--prediction-format", choices=("xyxy", "xywh"), default="xyxy")
    evaluate.add_argument("--ground-truth-format", choices=("xyxy", "xywh"), default="xywh")
    evaluate.add_argument("--output", type=Path)
    frames = commands.add_parser("extract-frames", help="Sample video frames and timestamps")
    frames.add_argument("--video", type=Path, required=True)
    frames.add_argument("--output", type=Path, required=True)
    frames.add_argument("--every", type=int, default=1)
    predict = commands.add_parser("predict", help="Run a trained two-output detector on images")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--weights", type=Path)
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--images", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "evaluate":
            report = evaluate_directories(
                args.predictions,
                args.ground_truth,
                threshold=args.iou,
                prediction_format=args.prediction_format,
                ground_truth_format=args.ground_truth_format,
            )
            result = json.dumps(report, indent=2, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(result + "\n", encoding="utf-8")
            print(result)
        elif args.command == "extract-frames":
            from .video import extract_frames

            count = extract_frames(args.video, args.output, args.every)
            logging.info("Saved %s frames to %s", count, args.output)
        elif args.command == "predict":
            from .config import DetectorConfig
            from .inference import predict_directory
            from .models import load_detector

            config = DetectorConfig.load(args.config)
            model = load_detector(args.model, args.weights)
            count = predict_directory(model, args.images, args.output, config)
            logging.info("Processed %s images; results saved to %s", count, args.output)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        logging.error("%s", error)
        return 1
    return 0
