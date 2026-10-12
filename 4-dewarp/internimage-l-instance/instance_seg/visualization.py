"""Instance mask overlays."""

from typing import Sequence, Tuple
import cv2
import numpy as np


COLORS = [(230, 80, 80), (80, 210, 120), (80, 140, 240), (235, 190, 65), (180, 90, 225)]

def render_mask(
    overlay: np.ndarray,
    mask: np.ndarray,
    bbox: Sequence[float],
    score: float,
    instance_id: int,
    color: Tuple[int, int, int],
) -> None:
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color_array).astype(np.uint8)
    x1, y1, x2, y2 = [float(value) for value in bbox]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 3)
    cv2.putText(
        overlay,
        f"doc {instance_id}: {score:.3f}",
        (round(x1), max(18, round(y1) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
