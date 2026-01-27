<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/images/logo_dark.png">
    <img src="docs/static/images/logo.png" alt="BRACE logo" width="300">
  </picture>
  <h1>When Replanning Becomes the Bottleneck: BRACE for Budgeted Embodied Agent Replanning</h1>
</div>

<a href="docs/"><img src="https://img.shields.io/badge/Paper-PDF-7C3AED.svg" alt="Paper"></a>
<a href="https://anonymous-2026.github.io/BRACE-ICML"><img src="https://img.shields.io/badge/Website-Docs-0EA5E9.svg" alt="Website"></a>
<a href="https://drive.google.com/drive/folders/11ONHDWuz-Es9sc9vvv5rQYVdMQgrBikl?usp=sharing"><img src="https://img.shields.io/badge/Demos-Google%20Drive-F97316.svg" alt="Demos"></a>
<a href="https://www.deepspeed.ai/"><img alt="DeepSpeed" src="https://img.shields.io/badge/DeepSpeed-0A66FF?logo=microsoft&logoColor=white"></a>
<a href="https://github.com/microsoft/AirSim"><img src="https://img.shields.io/badge/-AirSim-0066CC.svg" alt="Microsoft AirSim"></a>
<a href="https://aihabitat.org/"><img src="https://img.shields.io/badge/-Habitat-0052CC.svg" alt="Meta Habitat"></a>
<a href="https://github.com/MARS-EAI/RoboFactory"><img src="https://img.shields.io/badge/-RoboFactory-008B8B.svg" alt="RoboFactory"></a>

BRACE treats **high-frequency replanning** as a **systems bottleneck** for embodied agents: repeated replanning under context growth leads to tail latency and SLO violations. We integrate a budgeted replanning controller (BRACE) with auditable logging (phase cost breakdown) and composable efficiency modules (E-RECAP pruning, RAG/memory, budget-matched baselines), evaluated across multiple domains (Habitat, RoboFactory, AirSim) and a VLA executor track (OpenMARL).

## Showcases (short video loops)

| Habitat (tail/SLO) | RoboFactory (coordination) | AirSim (cinematic) |
|---|---|---|
| ![](docs/static/images/habitat_compare.gif) | ![](docs/static/images/robofactory_compare.gif) | ![](docs/static/images/airsim_compare.gif) |

Full MP4 clips (for the website) live in `docs/static/videos/` and are embedded in `docs/index.html`.

## Quick links

- Website (static template): `docs/index.html`
- Docs (single guide): `docs/README.md`
- Controller spec (proxy-ready): `docs/CONTROLLER.md`
- Logging schema: `docs/SCHEMA.md`
- Analysis utilities: `analysis/README.md`
- Demo/media provenance: `docs/PROVENANCE.md`
- Configs: `configs/smoke/` (defaults) and `configs/experiments/` (curated eval/demo configs)
- E-RECAP (v1) module: `docs/e-recap.md` / `e-recap/`
- Scripts (runnable entrypoints): `scripts/`

## Minimal smoke (no simulators)

```bash
scripts/smoke_local.sh
```

## Repo policy (open-source)

This repository is **code + configs + docs only**.

- Do not commit large weights/datasets/videos; keep them under your `BRACE_MODELS_ROOT` / `BRACE_DATA_ROOT` and reference via env vars.
- Local outputs are generated at runtime (e.g., `runs/`, `artifacts/`, `data/`) and are **not** shipped in this repo. See `docs/LOCAL_OUTPUTS.md`.
