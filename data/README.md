# `data/` (local manifests; optional)

Some utilities write small intermediate files under `data/` (git-ignored by default).

Example:
- `experiments/robofactory/instructions/generate_robofactory_instruction_manifest.py` defaults to
  `data/robofactory_instructions/*` for generated JSONL manifests.

Policy:
- keep large/generated manifests under `data/` (ignored)
- keep only human-written checklists/configs under version control (e.g., `*.md`, `*.json`, `*.yaml`)
