from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def extract_keyframes(
    *,
    video_path: Path,
    output_dir: Path,
    num_frames: int = 3,
    indices: Optional[List[int]] = None,
) -> List[Path]:
    """Extract a few representative PNG screenshots from an MP4.

    - If `indices` is provided, it is interpreted as absolute frame indices.
    - Otherwise, extracts `num_frames` frames roughly evenly spaced.

    Returns: list of written PNG paths (may be shorter than requested).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required to extract screenshots from MP4.") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            # Best-effort fallback: read sequentially and pick early frames.
            frame_count = 1

        if indices is None:
            k = max(1, int(num_frames))
            if k == 1:
                indices = [0]
            else:
                indices = [int(round(i * (frame_count - 1) / float(k - 1))) for i in range(k)]

        written: List[Path] = []
        for i, idx in enumerate(list(indices)):
            idx = max(0, int(idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            outp = output_dir / f"shot_{i:03d}__frame_{idx:06d}.png"
            ok2 = cv2.imwrite(str(outp), frame)
            if ok2:
                written.append(outp)
        return written
    finally:
        try:
            cap.release()
        except Exception:
            pass

