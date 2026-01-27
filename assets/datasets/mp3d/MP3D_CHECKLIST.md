# MP3D checklist (do not commit the dataset)

This repo does **not** distribute MP3D. This file is a reproducible checklist for local setup.

## Expected layout

- `assets/datasets/mp3d/` (local path; keep data outside git)

## Verification (example)

Run these locally and paste outputs into your internal logs (not into this repo):

```bash
# Example: locate a few scene files (adjust extension to your MP3D format)
find /path/to/mp3d -maxdepth 3 -type f | head
```
