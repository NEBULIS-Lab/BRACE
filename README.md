<div align="center">
  <h1 style="display: inline-flex; align-items: center; justify-content: center; gap: 10px; margin: 0; vertical-align: middle;">
    <picture style="display: inline-block; vertical-align: middle; line-height: 1;">
      <source media="(prefers-color-scheme: dark)" srcset="docs/static/images/logo_dark.svg">
      <img src="docs/static/images/logo.svg" alt="BRACE logo" width="50" style="display: block; vertical-align: middle; margin: 0;">
    </picture>
    <span style="display: inline-block; vertical-align: middle; line-height: 1.2;">When Replanning Becomes the Bottleneck: BRACE for Budgeted Embodied Agent Replanning</span>
  </h1>
</div>

<a href="#"><img src="https://img.shields.io/badge/Paper-ICML%202026-6366F1.svg" alt="Paper"></a>
<a href="#"><img src="https://img.shields.io/badge/Website-Docs-6366F1.svg" alt="Website"></a>
<a href="#"><img src="https://img.shields.io/badge/Demos-Video%20%2B%20Screenshots-FF6B35.svg" alt="Demos"></a>
<a href="#"><img src="https://img.shields.io/badge/Schema-Auditable-22C55E.svg" alt="Schema"></a>
![](https://img.shields.io/badge/PRs-Welcome-blue)

BRACE treats **high-frequency replanning** as a **systems bottleneck** for embodied agents: repeated replanning under context growth leads to tail latency and SLO violations. We integrate a budgeted replanning controller (BRACE) with auditable logging (phase cost breakdown) and composable efficiency modules (E-RECAP pruning, RAG/memory, budget-matched baselines), evaluated across multiple domains (Habitat, RoboFactory, AirSim) and a VLA executor track (OpenMARL).

## Showcases (short video loops)

| Habitat (tail/SLO) | RoboFactory (coordination) | AirSim (cinematic) |
|---|---|---|
| ![](docs/static/images/habitat_compare.gif) | ![](docs/static/images/robofactory_compare.gif) | ![](docs/static/images/airsim_compare.gif) |

Full MP4 clips (for the website) live in `docs/static/videos/` and are embedded in `docs/index.html`.

## Quick links

- Website (static template): `docs/index.html`
- Docs (single guide): `docs/README.md`
- Configs: `configs/smoke/` (defaults) and `configs/paper/` (paper/demos)
- E-RECAP module (lightweight subset): `erecap/`
- Scripts (runnable entrypoints): `scripts/`

## Repo policy (open-source)

This repository is **code + configs + docs only**.

- Do not commit large weights/datasets/videos; keep them under your `BRACE_MODELS_ROOT` / `BRACE_DATA_ROOT` and reference via env vars.
- Local runs/artifacts are generated under `runs/` and `artifacts/` (ignored by git).
