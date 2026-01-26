from __future__ import annotations

import os
from pathlib import Path


def airsim_documents_dir() -> Path:
    # AirSim defaults to "Documents/AirSim" (Linux: ~/Documents/AirSim).
    override = os.environ.get("AIRSIM_DOCUMENTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Documents" / "AirSim").resolve()


def settings_json_path() -> Path:
    override = os.environ.get("AIRSIM_SETTINGS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return airsim_documents_dir() / "settings.json"

