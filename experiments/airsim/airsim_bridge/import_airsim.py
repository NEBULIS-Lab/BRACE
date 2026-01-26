from __future__ import annotations

import sys
from pathlib import Path


def import_airsim():
    """Import `airsim` with a vendored fallback.

    Constraint: open-source version must not depend on local `UAV/ref/*` folders.
    Supported:
    - `pip install airsim` (if available)
    - vendored `UAV/HATTO-UFog/AirSim/vendor/airsim`
    """

    try:
        import airsim  # type: ignore

        return airsim
    except Exception:
        base_dir = Path(__file__).resolve().parents[1]  # .../AirSim
        vendor_dir = base_dir / "vendor"
        if vendor_dir.exists():
            sys.path.insert(0, str(vendor_dir))
        import airsim  # type: ignore

        return airsim

