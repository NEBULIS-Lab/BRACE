<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/images/logo_dark.png">
    <img src="docs/static/images/logo.png" alt="BRACE logo" width="300">
  </picture>
  <h1>When Replanning Becomes the Bottleneck: BRACE for Budgeted Embodied Agent Replanning</h1>
</div>

<a href="docs/"><img src="https://img.shields.io/badge/Docs-Guide-7C3AED.svg" alt="Docs"></a>
<a href="https://anonymous-2026.github.io/BRACE-ICML"><img src="https://img.shields.io/badge/Website-Page%20URL-0EA5E9.svg" alt="Website"></a>
<a href="https://drive.google.com/drive/folders/1Aafo36p2JB8cCeWyG9-QRmEWrssk4kXM?usp=sharing"><img src="https://img.shields.io/badge/Demos-Google%20Drive-F97316.svg" alt="Demos"></a>
<a href="https://www.deepspeed.ai/"><img alt="DeepSpeed" src="https://img.shields.io/badge/DeepSpeed-0A66FF?logo=microsoft&logoColor=white"></a>
<a href="https://github.com/microsoft/AirSim"><img src="https://img.shields.io/badge/AirSim-Microsoft-16A34A.svg" alt="Microsoft AirSim"></a>
<a href="https://aihabitat.org/"><img src="https://img.shields.io/badge/Habitat-Meta-DC2626.svg" alt="Meta Habitat"></a>
<a href="https://github.com/MARS-EAI/RoboFactory"><img src="https://img.shields.io/badge/-RoboFactory-0D9488.svg" alt="RoboFactory"></a>

**BRACE** = **B**udgeted **R**eplanning and **C**oordination for **E**mbodied **A**gents.

BRACE targets **high-frequency replanning** as a **systems bottleneck** for embodied agents: as context grows (history, perception summaries, retrieved memory), replanning latency develops a heavy tail and leads to **deadline/SLO misses**.

BRACE provides:
- A **budgeted replanning controller** (when to replan, what to include, and how to stay within a time/token budget), and
- **Auditable phase logging** (token + latency accounting on the replanning call path),
and composes with efficiency modules (e.g., **E-RECAP** token pruning, optional retrieval/RAG).

> **Note (double-blind):** This repository accompanies an ICML 2026 submission. Please avoid adding author-identifying information to public artifacts during the review period.

## Showcases (side-by-side demos)

Each showcase below is a small, GitHub-friendly loop. Full-resolution artifacts are in the public Google Drive folder (badge above).

### Microsoft AirSim — multi-agent intersection (K=8)

<img src="docs/static/images/airsim_compare.gif" style="max-width: 100%; height: auto;" />

Left: baseline. Right: BRACE.

- Short MP4: `docs/static/videos/airsim_compare.mp4`

### Meta AI Habitat — navigation under strict SLO (tail latency)

<img src="docs/static/images/habitat_compare.gif" style="max-width: 100%; height: auto;" />

Left: no-prune. Right: E-RECAP pruning.

- Short MP4: `docs/static/videos/habitat_compare.mp4`

### RoboFactory — multi-agent manipulation (coordination)

<img src="docs/static/images/robofactory_compare.gif" style="max-width: 100%; height: auto;" />

Left: baseline. Right: BRACE + E-RECAP.

- Short MP4: `docs/static/videos/robofactory_compare.mp4`

## Representative results

Numbers below are aggregated from paper-facing tables produced by the postprocess pipeline (see **Reproducibility & auditing**). We report **replanning** tail latency and **SLO violation rate** (fraction of replanning calls exceeding the per-platform SLO).

| Platform | Setting | Baseline → Method | Tokens after (mean) | Lat P95 (ms) | SLO viol. |
|---|---|---|---:|---:|---:|
| Meta AI Habitat | navigation (30 eps) | no-prune → E-RECAP prune \(r=0.7\) | 235.07 → **20.06** | 2676.85 → **2499** | 85.5% → **3.6%** |
| RoboFactory | PassShoe (10 eps, SLO=250ms) | no-BRACE → BRACE + E-RECAP \(r=0.7\) | 1566.37 → **318.73** | 1603.75 → **1213** | 100.0% → **50.0%** |
| Microsoft AirSim | K=8 intersection (10 eps) | baseline → BRACE | 2934.14 → **1113.59** | 8520.00 → **1640** | 100.0% → **4.7%** |

> **Note (how to read):** Success can saturate at 100% in easy regimes; BRACE is primarily evaluated on **tail/SLO behavior and auditable accounting** on the replanning call path.

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

## Quickstart (local smoke; no simulators required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Validates: run directory + schema + postprocess tables.
scripts/smoke_local.sh
```

> **Note (how notes work):** We use `> **Note:** ...` blocks throughout the README to call out constraints, caveats, or reviewer-facing guidance.

## Run (requires simulators / external assets)

These wrappers invoke the python entrypoints under `experiments/` and write results to `runs/`:

```bash
# Meta AI Habitat (requires local habitat-setup + MP3D license assets)
scripts/run_habitat.sh --config configs/smoke/habitat_setup.json --run-name habitat_smoke

# RoboFactory (requires RoboFactory runtime + assets)
scripts/run_robofactory.sh --config configs/smoke/robofactory_lift_barrier.json --run-name robofactory_smoke

# Microsoft AirSim (requires UE binaries + BRACE_AIRSIM_ENVS_ROOT)
scripts/run_airsim.sh --config configs/smoke/airsim_multidrone_demo.json --run-name airsim_demo --ue-env airsimnh
```

## Repo map (where to look)

- `brace/`: BRACE controller core (budgeting + stability mechanisms)
- `experiments/`: domain runners (Habitat / RoboFactory / AirSim / proxy + stubs)
- `analysis/`: aggregation + audit tools (tables, schema coverage, trigger/controller audits)
- `docs/`: project page + guides (incl. schema, controller spec, provenance)
- `configs/`: curated configs (`smoke/` for sanity checks; `experiments/` for paper-facing setups)
- `scripts/`: thin wrappers around Python entrypoints (smoke / run / postprocess)
- `e-recap/`: vendored E-RECAP module (optional)
- `third_party/openmarl/`: vendored OpenMARL components used by the VLA executor track

## Reproducibility & auditing (paper-facing tables)

This repo uses an on-disk, auditable run format:

- `runs/<run_id>/run.json`
- `runs/<run_id>/events.jsonl`
- `runs/<run_id>/episode_metrics.jsonl`

To postprocess a run into paper-facing tables:

```bash
scripts/postprocess_run.sh runs/<run_id>
```

This performs strict schema checks and writes markdown tables under `artifacts/tables/`. See:
- `docs/SCHEMA.md` for field definitions
- `docs/PROVENANCE.md` for committed demo media provenance

## Citation

> **Note:** Citation will be updated after acceptance.

## Repo policy (open-source)

This repository is **code + configs + docs only**.

- Do not commit large weights/datasets/videos; keep them under your `BRACE_MODELS_ROOT` / `BRACE_DATA_ROOT` and reference via env vars.
- Local outputs are generated at runtime (e.g., `runs/`, `artifacts/`, `data/`) and are **not** shipped in this repo. See `docs/LOCAL_OUTPUTS.md`.
