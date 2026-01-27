# BRACE guide (read this first)

This repository is **code-only**. It does **not** ship:
- large `runs/` / `artifacts/` outputs
- model weights / checkpoints
- licensed datasets (e.g., MP3D scenes/episodes)

You generate outputs locally and keep large assets outside git.

Website demo page:
- Open `docs/index.html` locally, or host `docs/` via GitHub Pages.
- Short MP4 clips are stored in `docs/static/videos/` (kept intentionally small).

Key docs:
- Logging schema: `docs/SCHEMA.md`
- Controller spec (proxy-ready): `docs/CONTROLLER.md`
- Analysis utilities: `analysis/README.md`
- Demo/media provenance: `docs/PROVENANCE.md`
- E-RECAP (v1) integration: `docs/e-recap.md`
- Local output dirs (`runs/`, `artifacts/`, `data/`): `docs/LOCAL_OUTPUTS.md`

---

## 0) Quick start (local smoke; no simulators required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Validates: run directory + schema + postprocess tables.
scripts/smoke_local.sh
```

---

## 1) Environment variables

- `BRACE_MODELS_ROOT`: LLM/VLM weights root
- `BRACE_DATA_ROOT`: optional general data root
- `BRACE_ROBOFACTORY_DATA_ROOT`: RoboFactory/OpenMARL runtime data root (checkpoints, caches)
- `BRACE_AIRSIM_ENVS_ROOT`: AirSim UE binaries root
- `BRACE_HABITAT_PY`: python executable for your Habitat env (only needed for the Habitat wrapper runner)

---

## 2) Run experiments (example entrypoints)

These scripts are thin wrappers around the python entrypoints under `experiments/`.

Configs are intentionally curated to keep this repo small:
- `configs/smoke/`: fast sanity checks (default paths for scripts/CLI)
- `configs/experiments/`: curated eval/demo configs (representative; extend as needed)

### Proxy / stub (no external simulators)

These two are intentionally dependency-free and are suitable for open-source smoke testing:

```bash
scripts/run_stub.sh --run-name stub_smoke --episodes 1
scripts/run_proxy.sh --run-name proxy_smoke
```

### E-RECAP (v1) module (optional)

E-RECAP code is vendored under `e-recap/`, but uses project-wide deps + checkpoints. See:
- `docs/e-recap.md`
- `scripts/run_e_recap_stage1.sh`, `scripts/run_e_recap_stage2.sh`, `scripts/run_e_recap_inference.sh`

### Habitat (navigation)

Requires your own `habitat-setup/` checkout + a working Habitat env.

```bash
scripts/run_habitat.sh --config configs/smoke/habitat_setup.json --run-name habitat_smoke
```

### RoboFactory (manipulation / multi-agent)

Requires RoboFactory runtime deps + assets.

Notes:
- OpenMARL policy code used by the VLA track is vendored under `third_party/openmarl/` (no separate OpenMARL checkout needed).
- Checkpoints/caches still live outside git (see `BRACE_ROBOFACTORY_DATA_ROOT` and `robofactory.run_dir` in configs).

```bash
scripts/run_robofactory.sh --config configs/smoke/robofactory_lift_barrier.json --run-name rf_smoke
```

### AirSim (vehicles / drones)

Requires local AirSim UE binaries and `BRACE_AIRSIM_ENVS_ROOT` set.

```bash
export BRACE_AIRSIM_ENVS_ROOT=/path/to/AirSim/envs
scripts/run_airsim.sh --config configs/smoke/airsim_multidrone_demo.json --run-name airsim_demo --ue-env airsimnh
```

---

## 3) Postprocess a run into paper-facing tables

Given `runs/<run_id>` exists in your workspace root:

```bash
scripts/postprocess_run.sh runs/<run_id>
```

This runs strict schema checks + aggregation + trigger/controller coverage and writes markdown tables under `artifacts/tables/` (append-only).
It also attempts domain-specific tables (RoboFactory/AirSim) when applicable; see `analysis/README.md` for the full list.

---

## 4) Datasets (what is intentionally missing)

- **MP3D** is license-gated and is not included. Keep any MP3D scenes/episodes outside git.
- See `assets/licenses/MP3D.md` and keep your local checklist under `assets/datasets/mp3d/MP3D_CHECKLIST.md`.
- The repo ships only configs/checklists; you must obtain datasets through their official channels.

---

## 5) Known issues / TODOs (paper-level)

These are **known gaps** for paper-level claims:

- **VLA-aware latency accounting**: supported when the runner logs `event_type="phase"` with `phase="vla_policy_call"`; `analysis/aggregate_runs.py` will then report VLA-aware control-loop latency.
- **RoboFactory SLO reporting**: a single tight SLO (e.g., 250ms) can saturate violations and hide improvements; prefer multi-threshold reporting.
- **BRACE (beyond pruning) in real domains**: proxy shows strong stability deltas; real-domain deltas still need strengthening.
