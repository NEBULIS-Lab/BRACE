#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GifSpec:
    seconds: float
    fps: float
    colors: int
    start_seconds: float = 0.0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    _ensure_parent(dst)
    shutil.copy2(src, dst)


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _video_info(path: Path) -> dict:
    cap = _open_video(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = (n / fps) if fps > 0 and n > 0 else 0.0
    return {"width": w, "height": h, "fps": fps, "frames": n, "duration_s": dur}


def _read_frames_rgb(video_path: Path, *, start_seconds: float, seconds: float, out_fps: float) -> list[np.ndarray]:
    info = _video_info(video_path)
    in_fps = info["fps"] if info["fps"] > 0 else out_fps
    step = max(1, int(round(in_fps / out_fps)))

    cap = _open_video(video_path)
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000.0)

    frames: list[np.ndarray] = []
    max_frames = max(1, int(round(seconds * out_fps)))

    i = 0
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        i += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def _make_palette_from_samples(frames_rgb: list[np.ndarray], *, colors: int) -> Image.Image:
    sample_count = min(8, len(frames_rgb))
    sample_idx = np.linspace(0, len(frames_rgb) - 1, sample_count).astype(int).tolist()

    downs: list[np.ndarray] = []
    for idx in sample_idx:
        f = frames_rgb[idx]
        h, w = f.shape[:2]
        target_w = 512
        if w > target_w:
            target_h = max(1, int(round(h * (target_w / w))))
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
        downs.append(f)

    tile = np.concatenate(downs, axis=1)
    pil = Image.fromarray(tile, mode="RGB")
    pal = pil.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
    return pal


def _encode_gif(
    frames_rgb: list[np.ndarray],
    *,
    out_path: Path,
    fps: float,
    colors: int,
) -> None:
    palette = _make_palette_from_samples(frames_rgb, colors=colors)

    frames_p: list[Image.Image] = []
    for f in frames_rgb:
        pil = Image.fromarray(f, mode="RGB")
        frames_p.append(pil.quantize(palette=palette, dither=Image.Dither.NONE))

    duration_ms = int(round(1000.0 / fps))
    _ensure_parent(out_path)
    frames_p[0].save(
        out_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _encode_side_by_side_mp4(
    *,
    left_mp4: Path,
    right_mp4: Path,
    out_mp4: Path,
    start_seconds: float = 0.0,
    seconds: float | None = None,
) -> None:
    cap_l = _open_video(left_mp4)
    cap_r = _open_video(right_mp4)

    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w_l, h_l) != (w_r, h_r):
        raise ValueError(f"Videos differ in size: {left_mp4}={w_l}x{h_l}, {right_mp4}={w_r}x{h_r}")

    fps = float(cap_l.get(cv2.CAP_PROP_FPS) or cap_r.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_parent(out_mp4)
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w_l + w_r, h_l))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    if start_seconds > 0:
        cap_l.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000.0)
        cap_r.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000.0)

    max_frames = None
    if seconds is not None and seconds > 0:
        max_frames = int(round(seconds * fps))

    try:
        written = 0
        while True:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            if not ok_l or not ok_r:
                break
            writer.write(np.hstack([frame_l, frame_r]))
            written += 1
            if max_frames is not None and written >= max_frames:
                break
    finally:
        cap_l.release()
        cap_r.release()
        writer.release()


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024.0 / 1024.0


def _render_gif_with_budget(
    *,
    video_path: Path,
    out_gif: Path,
    preferred: GifSpec,
    min_mb: float,
    max_mb: float,
) -> GifSpec:
    candidates: list[GifSpec] = []

    candidates.append(preferred)
    candidates.append(GifSpec(seconds=min(preferred.seconds, 4.0), fps=max(6.0, preferred.fps - 2.0), colors=max(64, preferred.colors - 32), start_seconds=preferred.start_seconds))
    candidates.append(GifSpec(seconds=min(preferred.seconds, 3.0), fps=max(5.0, preferred.fps - 4.0), colors=max(48, preferred.colors - 48), start_seconds=preferred.start_seconds))
    candidates.append(GifSpec(seconds=min(preferred.seconds, 2.5), fps=max(4.0, preferred.fps - 6.0), colors=max(32, preferred.colors - 64), start_seconds=preferred.start_seconds))

    best: GifSpec | None = None
    best_over_mb = float("inf")

    for spec in candidates:
        frames = _read_frames_rgb(video_path, start_seconds=spec.start_seconds, seconds=spec.seconds, out_fps=spec.fps)
        _encode_gif(frames, out_path=out_gif, fps=spec.fps, colors=spec.colors)
        size = _size_mb(out_gif)
        if min_mb <= size <= max_mb:
            return spec
        if size > max_mb and size < best_over_mb:
            best_over_mb = size
            best = spec

    if best is None:
        best = preferred

    frames = _read_frames_rgb(video_path, start_seconds=best.start_seconds, seconds=best.seconds, out_fps=best.fps)
    _encode_gif(frames, out_path=out_gif, fps=best.fps, colors=best.colors)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Render high-res showcase media for README + website (no ffmpeg).")
    parser.add_argument("--min-gif-mb", type=float, default=4.0)
    parser.add_argument("--max-gif-mb", type=float, default=10.0)
    parser.add_argument("--airsim-src", type=str, default="", help="Optional: source MP4 for AirSim side-by-side compare.")
    parser.add_argument("--habitat-src", type=str, default="", help="Optional: source MP4 for Habitat side-by-side compare.")
    parser.add_argument("--robofactory-baseline-src", type=str, default="", help="Optional: source MP4 for RoboFactory baseline.")
    parser.add_argument("--robofactory-method-src", type=str, default="", help="Optional: source MP4 for RoboFactory method.")
    parser.add_argument("--robofactory-start-seconds", type=float, default=8.0, help="Start time for RoboFactory compare clip.")
    parser.add_argument("--robofactory-seconds", type=float, default=3.0, help="Duration for RoboFactory compare clip.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    videos_dir = repo_root / "docs" / "static" / "videos"
    images_dir = repo_root / "docs" / "static" / "images"

    # Website MP4s (keep native resolution; do not resize).
    airsim_out_mp4 = videos_dir / "airsim_compare.mp4"
    habitat_out_mp4 = videos_dir / "habitat_compare.mp4"
    robofactory_out_mp4 = videos_dir / "robofactory_compare.mp4"

    if args.airsim_src:
        _copy_file(Path(args.airsim_src), airsim_out_mp4)
    if args.habitat_src:
        _copy_file(Path(args.habitat_src), habitat_out_mp4)
    if args.robofactory_baseline_src and args.robofactory_method_src:
        _encode_side_by_side_mp4(
            left_mp4=Path(args.robofactory_baseline_src),
            right_mp4=Path(args.robofactory_method_src),
            out_mp4=robofactory_out_mp4,
            start_seconds=float(args.robofactory_start_seconds),
            seconds=float(args.robofactory_seconds),
        )

    # README GIFs (keep video dimensions; tune fps/duration/colors to fit size budget).
    gif_jobs = [
        (
            airsim_out_mp4,
            images_dir / "airsim_compare.gif",
            GifSpec(seconds=3.0, fps=6.0, colors=64, start_seconds=0.0),
        ),
        (
            habitat_out_mp4,
            images_dir / "habitat_compare.gif",
            GifSpec(seconds=4.0, fps=8.0, colors=96, start_seconds=0.0),
        ),
        (
            robofactory_out_mp4,
            images_dir / "robofactory_compare.gif",
            GifSpec(seconds=3.0, fps=6.0, colors=64, start_seconds=0.0),
        ),
    ]

    for mp4, gif, spec in gif_jobs:
        chosen = _render_gif_with_budget(
            video_path=mp4,
            out_gif=gif,
            preferred=spec,
            min_mb=float(args.min_gif_mb),
            max_mb=float(args.max_gif_mb),
        )
        print(f"[gif] {gif.relative_to(repo_root)} <- {mp4.relative_to(repo_root)} | chosen={chosen} | size={_size_mb(gif):.2f}MB")

    for p in [airsim_out_mp4, habitat_out_mp4, robofactory_out_mp4]:
        if p.exists():
            info = _video_info(p)
            print(f"[mp4] {p.relative_to(repo_root)} {info['width']}x{info['height']} fps={info['fps']:.2f} dur={info['duration_s']:.2f}s size={_size_mb(p):.2f}MB")
        else:
            print(f"[mp4] missing: {p.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
