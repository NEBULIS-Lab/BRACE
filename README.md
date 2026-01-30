<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/images/logo_dark.png">
    <img src="docs/static/images/logo.png" alt="BRACE logo" width="300">
  </picture>
  <h1>When Replanning Becomes the Bottleneck: Budgeted Embodied Agent Replanning</h1>
</div>

<a href="docs/"><img src="https://img.shields.io/badge/Docs-Guide-7C3AED.svg" alt="Docs"></a>
<a href="https://anonymous-2026.github.io/BRACE-ICML"><img src="https://img.shields.io/badge/Website-Page%20URL-0EA5E9.svg" alt="Website"></a>
<a href="https://drive.google.com/drive/folders/1Aafo36p2JB8cCeWyG9-QRmEWrssk4kXM?usp=sharing"><img src="https://img.shields.io/badge/Demos-Google%20Drive-F97316.svg" alt="Demos"></a>
<a href="https://www.deepspeed.ai/"><img alt="DeepSpeed" src="https://img.shields.io/badge/DeepSpeed-0A66FF?logo=microsoft&logoColor=white"></a>
<a href="https://github.com/microsoft/AirSim"><img src="https://img.shields.io/badge/AirSim-Microsoft-16A34A.svg" alt="Microsoft AirSim"></a>
<a href="https://aihabitat.org/"><img src="https://img.shields.io/badge/Habitat-Meta-DC2626.svg" alt="Meta Habitat"></a>
<a href="https://github.com/MARS-EAI/RoboFactory"><img src="https://img.shields.io/badge/-RoboFactory-0D9488.svg" alt="RoboFactory"></a>

**BRACE** = **B**udgeted **R**eplanning **a**nd **C**oordination for **E**mbodied-agents.

This work targets **high-frequency replanning** as a **systems bottleneck** for embodied agents: as context grows (history, perception summaries, retrieved memory), replanning latency develops a heavy tail and leads to **deadline/SLO misses**.

BRACE provides:
- A **budgeted replanning controller** (when to replan, what to include, and how to stay within a time/token budget), and
- **Auditable phase logging** (token + latency accounting on the replanning call path),
and composes with efficiency modules (e.g., **E-RECAP** token pruning, optional retrieval/RAG).

> **Note (double-blind):** This repository accompanies an ICML 2026 submission. Please avoid adding author-identifying information to public artifacts during the review period.

## Showcases (side-by-side demos)

Each showcase below is a small, GitHub-friendly loop. Full-resolution artifacts are in the public Google Drive folder (badge above).

### Microsoft AirSim — multi-agent intersection

<img src="docs/static/images/airsim_compare.gif" style="max-width: 100%; height: auto;" />

**Scenario:** 8 drones navigating through a shared intersection with high-frequency replanning. **Left:** baseline (no budget control) suffers from replanning latency spikes, leading to coordination failures and collisions. **Right:** BRACE with budgeted replanning maintains stable, coordinated flight paths and avoids deadline misses.

- Short MP4: [docs/static/videos/airsim_compare.mp4](docs/static/videos/airsim_compare.mp4)

### Meta Habitat — navigation under strict SLO (tail latency)

<img src="docs/static/images/habitat_compare.gif" style="max-width: 100%; height: auto;" />

**Scenario:** PointGoal navigation task with strict SLO (2500ms per replanning call). **Left:** baseline (no pruning) exceeds SLO frequently, causing agent stalling. **Right:** E-RECAP token pruning reduces context size and keeps replanning within SLO, enabling smooth navigation.

- Short MP4: [docs/static/videos/habitat_compare.mp4](docs/static/videos/habitat_compare.mp4)

### RoboFactory — multi-agent manipulation (coordination)

<img src="docs/static/images/robofactory_compare.gif" style="max-width: 100%; height: auto;" />

**Scenario:** PassShoe task requiring tight coordination between two robots. **Left:** baseline suffers from deadlock and excessive wait times due to uncoordinated replanning. **Right:** BRACE + E-RECAP coordinates replanning decisions, reduces wait times, and maintains task success while staying within SLO.

- Short MP4: [docs/static/videos/robofactory_compare.mp4](docs/static/videos/robofactory_compare.mp4)

## Repo map

- **`brace/`**: BRACE controller core (budgeting + stability mechanisms)
- **`experiments/`**: domain runners (Habitat / RoboFactory / AirSim / proxy + stubs)
- **`analysis/`**: aggregation + audit tools (tables, schema coverage, trigger/controller audits)
  - Start here: [`analysis/README.md`](analysis/README.md)
- **`docs/`**: project page + guides
  - Website: [`docs/index.html`](docs/index.html)
  - Main guide: [`docs/README.md`](docs/README.md)
  - Controller spec: [`docs/CONTROLLER.md`](docs/CONTROLLER.md)
  - Logging schema: [`docs/SCHEMA.md`](docs/SCHEMA.md)
  - Demo/media provenance: [`docs/PROVENANCE.md`](docs/PROVENANCE.md)
  - E-RECAP guide: [`docs/e-recap.md`](docs/e-recap.md)
- **`configs/`**: curated configs
  - Defaults: `configs/smoke/` (sanity checks)
  - Paper-facing: `configs/experiments/` (curated eval/demo configs)
- **`scripts/`**: thin wrappers around Python entrypoints (smoke / run / postprocess)
- **`e-recap/`**: vendored E-RECAP module (optional)
- **`third_party/openmarl/`**: vendored OpenMARL components used by the VLA executor track


## Representative results

Numbers below are aggregated from paper-facing tables produced by the postprocess pipeline (see **Reproducibility & auditing**). We report **replanning** tail latency and **SLO violation rate** (fraction of replanning calls exceeding the per-platform SLO).

### Meta Habitat — navigation under strict SLO (30 eps, SLO=2500ms)

| Method | Success | Tokens after (mean) | Token reduction | Lat P95 (ms) | Lat P99 (ms) | SLO viol. |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (no pruning) | 100.0% | 235.07 | 0.0% | 2677 | 2700 | 85.5% |
| E-RECAP pruning (r=0.7) | 100.0% | **20.06** | **91.8%** | **2499** | 2533 | **3.6%** |
| BRACE (no pruning) | 100.0% | 234.68 | 0.0% | 2679 | 2688 | 85.3% |
| BRACE + E-RECAP (r=0.7) | 100.0% | **20.02** | **91.7%** | 2500 | **2504** | 4.7% |

### RoboFactory — PassShoe (10 eps, SLO=250ms)

| Method | Success | Tokens after (mean) | Token reduction | Lat P95 (ms) | Lat P99 (ms) | SLO viol. | Wait time mean (ms/ep) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 100.0% | 1566.37 | 0.0% | 1604 | 2481 | 100.0% | 9063 |
| E-RECAP pruning (r=0.7) | 100.0% | 350.45 | 77.6% | 1236 | 1239 | 100.0% | 7141 |
| Recency baseline | 100.0% | 350.37 | 77.7% | 1235 | 1246 | 100.0% | 7172 |
| BRACE (no pruning) | 100.0% | 1413.77 | 0.0% | 1587 | 1596 | **50.0%** | 4213 |
| BRACE + E-RECAP (r=0.7) | 100.0% | **318.73** | **77.4%** | **1213** | **1226** | **50.0%** | **3546** |

### Microsoft AirSim — multi-agent intersection (K=8, 10 eps, SLO=2500ms)

| Method | Success | Tokens after (mean) | Token reduction | Lat P50 (ms) | Lat P95 (ms) | Lat P99 (ms) | SLO viol. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 100.0% | 2934.14 | 0.0% | 5960 | 8520 | 9000 | 100.0% |
| BRACE + E-RECAP | 100.0% | **1113.59** | **65.0%** | **1640** | **1640** | 9120 | **4.7%** |

> **Note (how to read):** Success can saturate at 100% in easy regimes; BRACE is primarily evaluated on **tail/SLO behavior and auditable accounting** on the replanning call path.

## Quickstart (local smoke; no simulators required)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Validates: run directory + schema + postprocess tables.
scripts/smoke_local.sh
```

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
