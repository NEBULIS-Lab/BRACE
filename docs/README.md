# BRACE guide (read this first)

This repository is **code-only**. It does **not** ship:
- large `runs/` / `artifacts/` outputs
- model weights / checkpoints
- licensed datasets (e.g., MP3D scenes/episodes)

You generate outputs locally and keep large assets outside git.

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
- `BRACE_AIRSIM_ENVS_ROOT`: AirSim UE binaries root (Domain C)
- `BRACE_HABITAT_PY`: python executable for your Habitat env (Domain A)

---

## 2) Run experiments (example entrypoints)

These scripts are thin wrappers around the python entrypoints under `experiments/`.

### Domain A (Habitat)

Requires your own `habitat-setup/` checkout + a working Habitat env.

```bash
scripts/run_domain_a_habitat.sh --config configs/habitat_setup_smoke.json --run-name habitat_smoke
```

### Domain B (RoboFactory)

Requires your own RoboFactory workspace + assets; see `configs/robofactory/*.json` for expected knobs.

```bash
scripts/run_domain_b_robofactory.sh --config configs/robofactory/rf_table_lift_barrier_smoke.json --run-name rf_smoke
```

### Domain C (AirSim)

Requires local AirSim UE binaries and `BRACE_AIRSIM_ENVS_ROOT` set.

```bash
export BRACE_AIRSIM_ENVS_ROOT=/path/to/AirSim/envs
scripts/run_domain_c_airsim.sh --config configs/airsim/domainc_multidrone_demo.json --run-name airsim_demo --ue-env airsimnh
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

- **VLA-aware latency accounting**: current Domain B VLA tables may exclude `phase=vla_policy_call` from end-to-end latency totals.
- **Domain B SLO reporting**: a single tight SLO (e.g., 250ms) can saturate violations and hide improvements; prefer multi-threshold reporting.
- **BRACE (beyond pruning) in real domains**: proxy shows strong stability deltas; real-domain deltas still need strengthening.
