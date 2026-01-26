<div align="center">
  <h1 style="display: inline-flex; align-items: center; gap: 10px; margin: 0;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/site/static/images/logo_dark.svg">
      <img src="docs/site/static/images/logo.svg" alt="BRACE logo" width="50" style="vertical-align: middle;">
    </picture>
    BRACE: When Replanning Becomes the Bottleneck
  </h1>
</div>

<a href="#"><img src="https://img.shields.io/badge/Paper-ICML%202026-6366F1.svg" alt="Paper"></a>
<a href="#"><img src="https://img.shields.io/badge/Website-Docs-6366F1.svg" alt="Website"></a>
<a href="#"><img src="https://img.shields.io/badge/Demos-Video%20%2B%20Screenshots-FF6B35.svg" alt="Demos"></a>
<a href="#"><img src="https://img.shields.io/badge/Schema-Auditable-22C55E.svg" alt="Schema"></a>
![](https://img.shields.io/badge/PRs-Welcome-blue)

BRACE treats **high-frequency replanning** as a **systems bottleneck** for embodied agents: repeated replanning under context growth leads to tail latency and SLO violations. We integrate a budgeted replanning controller (BRACE) with auditable logging (phase cost breakdown) and composable efficiency modules (E-RECAP pruning, RAG/memory, budget-matched baselines), evaluated across multiple domains (Habitat, RoboFactory, AirSim) and a VLA executor track (OpenMARL).

## Showcases (representative frames)

| Habitat (Domain A) | RoboFactory (Domain B) | AirSim (Domain C) |
|---|---|---|
| ![](docs/site/static/images/habitat_demo_frame.png) | ![](docs/site/static/images/robofactory_demo_frame.png) | ![](docs/site/static/images/airsim_demo_panel.png) |

## Quick links

- Website (static template): `docs/site/index.html`
- Docs (single guide): `docs/README.md`
- Scripts (runnable entrypoints): `scripts/`

## Repo policy (open-source)

This repository is **code + configs + docs only**.

- Do not commit large weights/datasets/videos; keep them under your `BRACE_MODELS_ROOT` / `BRACE_DATA_ROOT` and reference via env vars.
- Local runs/artifacts are generated under `runs/` and `artifacts/` (ignored by git).
