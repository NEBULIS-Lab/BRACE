# E-RECAP (v1) module — vendored code

E-RECAP (Embodied REplanning with Cost-Aware Pruning) is an optional module used by BRACE to **accelerate high-frequency replanning** by pruning planner context in a **cost-aware** way.

## What it does (at a glance)

- Learns token importance priors and uses them to guide pruning.
- Applies dynamic, cost-aware pruning during replanning to reduce long-context overhead.
- Integrates as a planner-side optimization (drop-in; no need to change task definitions/environments/control policies).

## Quick start (BRACE-level)

Use the BRACE integration guide and wrappers:

- Guide: `docs/e-recap.md`
- Scripts: `scripts/run_e_recap_stage1.sh`, `scripts/run_e_recap_stage2.sh`, `scripts/run_e_recap_inference.sh`

## Repo policy (open-source)

- **No weights / checkpoints / datasets** are committed under `e-recap/`.
- Put model checkpoints under the project-level `checkpoints/` (git-ignored) and use `BRACE_MODELS_ROOT` / `BRACE_DATA_ROOT` as needed.
- Dependencies follow the project-wide `requirements.txt` (install PyTorch separately with the right CUDA build).

## Code layout (vendored)

- `e-recap/src/`: python entrypoints and modules
- `e-recap/src/multi_agent/`: cooperative multi-agent replanning utilities
- `e-recap/src/evaluation/`: evaluation harnesses / benchmarks

## Attribution / citation

This directory vendors the upstream E-RECAP codebase as a module inside BRACE. Please cite BRACE using `CITATION.cff` (or `docs/CITATION.cff`) when appropriate.