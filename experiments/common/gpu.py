from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class GpuInfo:
    index: int
    mem_used_mb: int
    util_pct: int


def _query_gpus() -> List[GpuInfo]:
    """Return a best-effort snapshot of GPUs via nvidia-smi.

    Expected CSV line format (no units): "<idx>, <mem_used_mb>, <util_pct>".
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return []

    gpus: List[GpuInfo] = []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            mem = int(parts[1])
            util = int(parts[2])
        except Exception:
            continue
        gpus.append(GpuInfo(index=idx, mem_used_mb=mem, util_pct=util))
    return gpus


def pick_idle_gpu(
    *,
    max_mem_used_mb: int = 1024,
    max_util_pct: int = 10,
    prefer_high_index: bool = True,
) -> Optional[int]:
    """Pick an "idle" GPU (low mem + low util), preferring higher index.

    If no GPU matches the idle thresholds, fall back to the least-loaded GPU
    (min util, then min mem), still tie-breaking by index preference.
    """
    gpus = _query_gpus()
    if not gpus:
        return None

    def tie(idx: int) -> int:
        return idx if prefer_high_index else -idx

    idle = [
        g
        for g in gpus
        if int(g.mem_used_mb) <= int(max_mem_used_mb) and int(g.util_pct) <= int(max_util_pct)
    ]
    if idle:
        best = sorted(idle, key=lambda g: (g.util_pct, g.mem_used_mb, -tie(g.index)))[0]
        return int(best.index)

    best = sorted(gpus, key=lambda g: (g.util_pct, g.mem_used_mb, -tie(g.index)))[0]
    return int(best.index)


def resolve_cuda_visible_devices(value, *, max_mem_used_mb: int = 1024, max_util_pct: int = 10) -> Optional[str]:
    """Resolve config-provided cuda_visible_devices to a CUDA_VISIBLE_DEVICES string.

    Supports:
      - None: no-op
      - int/float: treated as a single GPU id
      - str "auto": pick an idle GPU (prefers higher index)
      - str: passed through as-is (e.g., "3" or "3,4")
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        if v.lower() == "auto":
            picked = pick_idle_gpu(max_mem_used_mb=max_mem_used_mb, max_util_pct=max_util_pct, prefer_high_index=True)
            return str(picked) if picked is not None else None
        return v
    return str(value)

