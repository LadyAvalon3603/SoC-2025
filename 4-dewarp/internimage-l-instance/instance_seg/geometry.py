"""Mask geometry helpers used by single-pass deployment inference."""

from typing import Optional, Tuple

import cv2
import numpy as np


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an exclusive XYXY box tightly enclosing a boolean mask."""
    x, y, width, height = cv2.boundingRect(
        np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    )
    if width <= 0 or height <= 0:
        return None
    return x, y, x + width, y + height
