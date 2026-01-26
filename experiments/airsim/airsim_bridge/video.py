from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2


def _try_get_ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return str(exe)
    except Exception:
        return None
    return None


def _sorted_pngs(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png"))


def _read_frame(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _ensure_same_height(left, right):
    if left.shape[0] == right.shape[0]:
        return left, right
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h), interpolation=cv2.INTER_AREA)
    return left, right


def _encode_png_sequence_to_mp4(*, frames_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _try_get_ffmpeg_exe()
    if ffmpeg:
        import subprocess

        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return

    # Fallback: OpenCV mp4v.
    first = cv2.imread(str(next(iter(sorted(frames_dir.glob("frame_*.png"))))), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"No readable frames under: {frames_dir}")
    h, w = int(first.shape[0]), int(first.shape[1])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    try:
        for p in _sorted_pngs(frames_dir):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


def build_video_from_frames(*, frames_dir: Path, output_path: Path, fps: float) -> Tuple[int, int]:
    """Encode an MP4 from a single PNG sequence dir (frame_%06d.png).

    Returns: (num_frames_written, width_px)
    """
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found under: {frames_dir}")
    # Build a contiguous tmp sequence (skip corrupted frames).
    tmp_dir = output_path.parent / "_tmp_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    first = None
    written = 0
    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if first is None:
            first = img
        outp = tmp_dir / f"frame_{written:06d}.png"
        cv2.imwrite(str(outp), img)
        written += 1

    if first is None or written <= 0:
        raise RuntimeError(f"Failed to read any frames under: {frames_dir}")

    _encode_png_sequence_to_mp4(frames_dir=tmp_dir, output_path=output_path, fps=float(fps))

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return written, int(first.shape[1])


def build_side_by_side_video(
    *,
    frames_left_dir: Path,
    frames_right_dir: Path,
    output_path: Path,
    fps: float,
) -> Tuple[int, int]:
    """Create one MP4 with left/right videos concatenated (expects two PNG sequences)."""
    left_paths = _sorted_pngs(frames_left_dir)
    right_paths = _sorted_pngs(frames_right_dir)
    if not left_paths or not right_paths:
        raise RuntimeError("No frames found (need both left/right PNG sequences).")
    n = min(len(left_paths), len(right_paths))

    tmp_dir = output_path.parent / "_tmp_hconcat_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    first_frame = None
    frames_written = 0
    for i in range(n):
        try:
            left = _read_frame(left_paths[i])
            right = _read_frame(right_paths[i])
        except Exception:
            continue
        left, right = _ensure_same_height(left, right)
        frame = cv2.hconcat([left, right])
        if first_frame is None:
            first_frame = frame
        out_path = tmp_dir / f"frame_{frames_written:06d}.png"
        cv2.imwrite(str(out_path), frame)
        frames_written += 1

    if first_frame is None or frames_written <= 0:
        raise RuntimeError("Failed to find any readable left/right frame pairs for encoding.")

    _encode_png_sequence_to_mp4(frames_dir=tmp_dir, output_path=output_path, fps=float(fps))

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return frames_written, int(first_frame.shape[1])


def build_grid_video(
    *,
    frames_dirs: List[Path],
    grid_wh: Tuple[int, int],
    output_path: Path,
    fps: float,
) -> Tuple[int, int]:
    """Create one MP4 by tiling multiple sequences into a fixed grid."""
    gx, gy = int(grid_wh[0]), int(grid_wh[1])
    if gx <= 0 or gy <= 0:
        raise ValueError("grid_wh must be positive.")
    if len(frames_dirs) != gx * gy:
        raise ValueError(f"frames_dirs must have exactly {gx*gy} entries for grid {grid_wh}.")

    seqs = [_sorted_pngs(d) for d in frames_dirs]
    if any(not s for s in seqs):
        raise RuntimeError("One or more frame sequences are empty.")
    n = min(len(s) for s in seqs)

    tmp_dir = output_path.parent / "_tmp_grid_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    first_frame = None
    frames_written = 0
    for i in range(n):
        imgs = []
        ok = True
        for s in seqs:
            try:
                imgs.append(_read_frame(s[i]))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        # Normalize heights within each row.
        rows = []
        idx = 0
        for _r in range(gy):
            row_imgs = imgs[idx : idx + gx]
            idx += gx
            # match height across row
            h = min(im.shape[0] for im in row_imgs)
            row_imgs2 = []
            for im in row_imgs:
                if im.shape[0] != h:
                    row_imgs2.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA))
                else:
                    row_imgs2.append(im)
            rows.append(cv2.hconcat(row_imgs2))
        # Normalize widths across rows.
        w = min(r.shape[1] for r in rows)
        rows2 = []
        for r in rows:
            if r.shape[1] != w:
                rows2.append(cv2.resize(r, (w, int(r.shape[0] * w / r.shape[1])), interpolation=cv2.INTER_AREA))
            else:
                rows2.append(r)
        frame = cv2.vconcat(rows2)
        if first_frame is None:
            first_frame = frame
        out_path = tmp_dir / f"frame_{frames_written:06d}.png"
        cv2.imwrite(str(out_path), frame)
        frames_written += 1

    if first_frame is None or frames_written <= 0:
        raise RuntimeError("Failed to build any grid frames for encoding.")

    _encode_png_sequence_to_mp4(frames_dir=tmp_dir, output_path=output_path, fps=float(fps))

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return frames_written, int(first_frame.shape[1])


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "open_failed"}
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ok, _ = cap.read()
        return {"ok": bool(ok), "w": w, "h": h, "frames": n, "fps": fps}
    finally:
        cap.release()


def build_split_video(
    *,
    frames_left_dir: Path,
    frames_right_dir: Path,
    output_path: Path,
    fps: float,
) -> Tuple[int, int]:
    """Create one MP4 with left/right views concatenated.

    Returns: (num_frames_written, width_px)
    """

    left_paths = _sorted_pngs(frames_left_dir)
    right_paths = _sorted_pngs(frames_right_dir)
    if not left_paths or not right_paths:
        raise RuntimeError("No frames found (need both left/right PNG sequences).")
    n = min(len(left_paths), len(right_paths))

    # Some runs may contain a few empty/corrupted frames or missing indices. To keep encoding robust
    # and high-quality, we first build a *contiguous* split-frame sequence, then encode it.
    tmp_dir = output_path.parent / "_tmp_split_frames"
    if tmp_dir.exists():
        for p in tmp_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    first_frame = None
    frames_written = 0
    for i in range(n):
        try:
            left = _read_frame(left_paths[i])
            right = _read_frame(right_paths[i])
        except Exception:
            continue
        left, right = _ensure_same_height(left, right)
        frame = cv2.hconcat([left, right])
        if first_frame is None:
            first_frame = frame
        out_path = tmp_dir / f"frame_{frames_written:06d}.png"
        cv2.imwrite(str(out_path), frame)
        frames_written += 1

    if first_frame is None or frames_written <= 0:
        raise RuntimeError("Failed to find any readable left/right frame pairs for video encoding.")

    h = int(first_frame.shape[0])
    w = int(first_frame.shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _try_get_ffmpeg_exe()
    if ffmpeg:
        import subprocess

        # H.264 via ffmpeg (bundled via imageio-ffmpeg): substantially better quality than OpenCV mp4v.
        # Use CRF mode (visually lossless-ish) to avoid "blurry" textures on complex scenes.
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    else:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
        try:
            for i in range(frames_written):
                img = _read_frame(tmp_dir / f"frame_{i:06d}.png")
                writer.write(img)
        finally:
            writer.release()

    for p in tmp_dir.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    return frames_written, w
