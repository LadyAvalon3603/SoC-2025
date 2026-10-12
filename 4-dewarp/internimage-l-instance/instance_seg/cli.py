"""Argument parsing and validation without importing the model runtime."""

import argparse
from pathlib import Path

from . import settings


def parse_args(argv=None, *, output_mode=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run single-pass InternImage-L + MaskDINO instance inference with "
            "fixed-square Letterbox input configured by INPUT.IMAGE_SIZE. "
            "With no arguments, process the "
            "4-dewarp/TEST_dewarp directory."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=settings.DEFAULT_INPUT,
        help=f"Image file or directory (default: {settings.DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=settings.DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {settings.DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--config-file", type=Path, default=settings.DEFAULT_CONFIG)
    parser.add_argument(
        "--score-threshold", type=float, default=settings.DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=settings.DEFAULT_MAX_DETECTIONS,
        help=(
            "Maximum confidence-ranked instance masks restored for the "
            "candidate pool (default: 5)."
        ),
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=settings.DEFAULT_MIN_MASK_AREA,
        help=(
            "An instance mask's largest connected component must contain more "
            "than this many original-image pixels to enter the candidate pool."
        ),
    )
    parser.add_argument(
        "--amp",
        dest="use_amp",
        action="store_true",
        help="Use CUDA AMP/FP16 for MaskDINO inference (default).",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Use FP32 for MaskDINO inference.",
    )
    parser.add_argument("--output-dir", type=Path, default=settings.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-mode",
        choices=("debug", "production"),
        default=settings.DEFAULT_OUTPUT_MODE if output_mode is None else output_mode,
        help=(
            "debug saves final overlays; production saves only final "
            "full-resolution uint8 instance masks (0/255)."
        ),
    )
    parser.set_defaults(use_amp=settings.DEFAULT_USE_AMP)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process only the first N images; 0 processes all.",
    )
    args = parser.parse_args(argv)

    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    if args.min_mask_area < 0:
        raise ValueError("--min-mask-area must be non-negative")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    if args.output_mode not in ("debug", "production"):
        raise ValueError("OUTPUT_MODE must be debug or production")
    return args
