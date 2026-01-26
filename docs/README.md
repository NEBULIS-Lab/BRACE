# BRACE guide (read this first)

This repository is **code-only**. It does **not** ship:
- large `runs/` / `artifacts/` outputs
- model weights / checkpoints
- licensed datasets (e.g., MP3D scenes/episodes)

You generate outputs locally and keep large assets outside git.

Website demo page:
- Open `docs/index.html` locally, or host `docs/` via GitHub Pages.
- Short MP4 clips are stored in `docs/static/videos/` (kept intentionally small).

---

## 0) Quick start (analysis-only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Environment variables (no hard-coded paths)

- `BRACE_MODELS_ROOT`: LLM/VLM weights root
- `BRACE_DATA_ROOT`: optional general data root
- `BRACE_ROBOFACTORY_DATA_ROOT`: RoboFactory/OpenMARL data root (checkpoints, caches)
- `BRACE_AIRSIM_ENVS_ROOT`: AirSim UE binaries root
- `BRACE_HABITAT_PY`: python executable for your Habitat env

---

## 2) Run experiments (example entrypoints)

These scripts are thin wrappers around the python entrypoints under `experiments/`.

Configs are intentionally curated to keep this repo small:
- `configs/smoke/`: fast sanity checks (default paths for scripts/CLI)
- `configs/paper/`: paper-scale and demo configs (representative; extend as needed)

### Habitat (navigation)

Requires your own `habitat-setup/` checkout + a working Habitat env.

```bash
scripts/run_domain_a_habitat.sh --config configs/smoke/habitat_setup.json --run-name habitat_smoke
```

### RoboFactory (manipulation / multi-agent)

Requires your own RoboFactory workspace + assets; see `configs/smoke/` / `configs/paper/` for expected knobs.

```bash
scripts/run_domain_b_robofactory.sh --config configs/smoke/robofactory_lift_barrier.json --run-name rf_smoke
```

### AirSim (vehicles / drones)

Requires local AirSim UE binaries and `BRACE_AIRSIM_ENVS_ROOT` set.

```bash
export BRACE_AIRSIM_ENVS_ROOT=/path/to/AirSim/envs
scripts/run_domain_c_airsim.sh --config configs/smoke/airsim_multidrone_demo.json --run-name airsim_demo --ue-env airsimnh
```

---

## 3) Postprocess a run into paper-facing tables

Given `runs/<run_id>` exists in your workspace root:

```bash
scripts/postprocess_run.sh runs/<run_id>
```

This runs strict schema checks + aggregation + trigger/controller coverage and writes markdown tables under `artifacts/tables/` (append-only).

---

## 4) Datasets (what is intentionally missing)

- **MP3D** is license-gated and is not included. Keep any MP3D scenes/episodes outside git.
- The repo ships only configs and checks; you must obtain datasets through their official channels.

---

## 5) Known issues / TODOs (paper-level)

These are **known gaps** for paper-level claims:

- **VLA-aware latency accounting**: current RoboFactory VLA tables may exclude `phase=vla_policy_call` from end-to-end latency totals.
- **RoboFactory SLO reporting**: a single tight SLO (e.g., 250ms) can saturate violations and hide improvements; prefer multi-threshold reporting.
- **BRACE (beyond pruning) in real domains**: proxy shows strong stability deltas; real-domain deltas still need strengthening.
