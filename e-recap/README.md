# E-RECAP (v1) module — vendored code

This folder vendors the **E-RECAP** codebase (training + inference + evaluation) as a module inside BRACE.

Project-level policy:
- **No weights / checkpoints / datasets** are committed under `e-recap/`.
- Use the **project-wide** `checkpoints/` folder (git-ignored) and `BRACE_MODELS_ROOT`/`BRACE_DATA_ROOT` env vars.
- Use the **project-wide** `requirements.txt` for dependencies (Torch must be installed separately with the right CUDA build).

Entry docs & scripts (BRACE-level):
- Guide: `docs/e-recap.md`
- Scripts: `scripts/run_e_recap_*.sh`

Code layout (vendored):
- `e-recap/src/`: python entrypoints and modules
