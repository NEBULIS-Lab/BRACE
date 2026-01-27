# MP3D license notes (placeholder)

MP3D is **license-gated**. This repository does **not** distribute MP3D data.

## Policy
- Store MP3D data under `assets/datasets/mp3d/` (or equivalent shared storage) but never commit the dataset itself.
- Keep a reproducible checklist and verification commands in:
  - `assets/datasets/mp3d/MP3D_CHECKLIST.md`

## What to record in standup logs when MP3D is added
- Who acquired it (license holder), acquisition date, and where it was placed (absolute path).
- The verification commands used and their outputs (at least one `find ... -name '*.glb'` hit).

