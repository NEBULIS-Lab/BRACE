from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


ColorBGR = Tuple[int, int, int]


@dataclass(frozen=True)
class OverlayLine:
    text: str
    color: ColorBGR = (255, 255, 255)
    scale: float = 1.0
    thickness: int = 1


def draw_overlay_panel(
    img,
    lines: Iterable[OverlayLine],
    *,
    origin_xy: Tuple[int, int] = (18, 32),
    line_h: int = 19,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.47,
    thickness: int = 1,
    outline_thickness: int = 3,
    panel_padding: int = 10,
    panel_alpha: float = 0.38,
):
    """Draw a readable overlay block (multi-color lines + translucent background)."""

    if img is None:
        return img

    lines = [ln for ln in lines if ln.text]
    if not lines:
        return img

    x0, y0 = origin_xy
    sizes = [
        cv2.getTextSize(ln.text, font_face, float(font_scale) * float(ln.scale), max(1, int(thickness)))[0] for ln in lines
    ]
    max_w = max(w for (w, _h) in sizes)
    max_h = max(h for (_w, h) in sizes)
    block_w = max_w + 2 * panel_padding
    block_h = len(lines) * line_h + 2 * panel_padding

    x1 = max(0, x0 - panel_padding)
    y1 = max(0, y0 - max_h - panel_padding)
    x2 = min(img.shape[1] - 1, x1 + block_w)
    y2 = min(img.shape[0] - 1, y1 + block_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, float(panel_alpha), img, 1.0 - float(panel_alpha), 0, dst=img)

    for i, ln in enumerate(lines):
        y = y0 + i * line_h
        sc = float(font_scale) * float(ln.scale)
        th = max(1, int(ln.thickness)) if ln.thickness else max(1, int(thickness))
        cv2.putText(img, ln.text, (x0, y), font_face, sc, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(img, ln.text, (x0, y), font_face, sc, ln.color, th, cv2.LINE_AA)

    return img


def decode_png(png_bytes: bytes):
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_png(path, img) -> None:
    cv2.imwrite(str(path), img)
