# Demo & Media Provenance (public)

This file tracks provenance metadata for the **small** media assets committed under `docs/static/`.
The goal is to make demo selection auditable without committing large raw artifacts.

Policy:
- Keep committed demo media **small** (GitHub Pages friendly).
- Record a stable checksum (`sha256`) for each committed asset.
- When available, record the **source run/config** used to produce the full-resolution artifact (kept outside this repo).

## `docs/static/videos/` (MP4)

| Asset | Size | sha256 | Source run/config (external; fill when known) |
|---|---:|---|---|
| `docs/static/videos/habitat_compare.mp4` | 582KB (595821B) | `84e631d60881744e1791f6d7db90570ad465cb97e6397823093062c321f9f9d1` | TBD |
| `docs/static/videos/robofactory_compare.mp4` | 544KB (556771B) | `b57991cf0341fb17acd1d0c04ffe6fd7cb6ff545ce8301d6c9019f201323d57b` | TBD |
| `docs/static/videos/airsim_compare.mp4` | 357KB (365360B) | `e14d8f1de694df976a5dded4b414522dea35e7e089e1b00220df63770eb301ef` | TBD |

## `docs/static/images/` (GIF/PNG)

| Asset | Size | sha256 | Source run/config (external; fill when known) |
|---|---:|---|---|
| `docs/static/images/habitat_compare.gif` | 3.0MB (3137852B) | `69c71945f7e0f16dda1c13b998e4c1d7077190dbd265606b2c0f29d981ca0a06` | TBD |
| `docs/static/images/robofactory_compare.gif` | 2.4MB (2418880B) | `2d2498ece26cca7bcec2355a402cd2d7c8d5cf6633589da9a32cffc4ed1fa7b7` | TBD |
| `docs/static/images/airsim_compare.gif` | 3.1MB (3194570B) | `b4d392ebe11452c6ff614c798f3f2b86e19bada1a28cb9d7ac383a43458f5f3a` | TBD |
| `docs/static/images/habitat_demo_frame.png` | 924KB (945737B) | `fd8a9b8954350d1993e0b1d11d6f415f38871f2435dc9f4dc1b521694ce3bcdb` | TBD |
| `docs/static/images/robofactory_demo_frame.png` | 1019KB (1042487B) | `54a5b7a34a68ed8ba54083484a91b63ab4eaeface573b81d059890ee7fc210d5` | TBD |
| `docs/static/images/airsim_demo_panel.png` | 2.3MB (2332739B) | `75f16178fad47fe0c0a7252484ca969feaec7bdc4614a699945bd3eaaa0973a5` | TBD |

## How to update this file

When replacing any committed media:
1) Keep filenames stable when possible (avoid breaking `docs/index.html` references).
2) Update the `Size` + `sha256` fields above.
3) Fill `Source run/config` if you have the generating run id, config JSON, and any postprocess table path.

