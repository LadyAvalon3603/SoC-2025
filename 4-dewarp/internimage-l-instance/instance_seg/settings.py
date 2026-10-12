"""Deployment defaults. CLI arguments override these values."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "internimage_l_maskdino.yaml"

# IDE-friendly defaults.  Paths are based on this file rather than the IDE's
# working directory, so infer.py can be launched directly from PyCharm.
DEFAULT_INPUT = PACKAGE_ROOT.parent / "TEST_dewarp"
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints" / "best_hd95_1024.pth"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "outputs"
DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_USE_AMP = True
# Restore at most five confidence-ranked instance masks.  A mask must contain
# more than 100 foreground pixels after retaining its largest connected region
# at original-image resolution; only the largest surviving instance is written.
DEFAULT_MAX_DETECTIONS = 5
DEFAULT_MIN_MASK_AREA = 100
DEFAULT_OUTPUT_MODE = "debug"
