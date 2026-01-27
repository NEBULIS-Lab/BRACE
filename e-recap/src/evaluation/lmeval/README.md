# Phase D: lm-eval-harness Integration for E-RECAP

This module provides custom task and model wrappers for evaluating E-RECAP models using the lm-eval-harness framework.

## ⚠️ Important: Setup Phase Only

**This is a setup/placeholder implementation:**
- ✅ Does NOT execute real inference
- ✅ Does NOT load any actual model
- ✅ Does NOT require GPU
- ✅ Only defines structure, classes, and method signatures
- ✅ Safe to run for validation and setup verification

## 📁 Directory Structure

```
src/evaluation/lmeval/
├── __init__.py              # Package initialization
├── longbench_task.py        # Custom LongBench task for lm-eval-harness
├── erecap_model.py            # E-RECAP model wrapper for lm-eval-harness
├── run_lmeval.py            # Main execution script (no inference)
├── longbench.yaml           # Task configuration template
└── README.md                # This file
```

## 🚀 Quick Start

### Basic Setup (No Inference)

```bash
# Run setup for a LongBench task
python3 src/evaluation/lmeval/run_lmeval.py \
    --task_config data/LongBench/narrativeqa.json \
    --model_name checkpoints/qwen2-7b-instruct \
    --output results/lmeval_narrativeqa_baseline_setup.json
```

### With E-RECAP Pruning Module

```bash
# Run setup with E-RECAP pruning
python3 src/evaluation/lmeval/run_lmeval.py \
    --task_config data/LongBench/narrativeqa.json \
    --model_name checkpoints/qwen2-7b-instruct \
    --pruner checkpoints/pruning_module.pt \
    --output results/lmeval_narrativeqa_erecap_setup.json
```

### Using Script

```bash
# Baseline setup
bash scripts/run_lmeval_setup.sh data/LongBench/narrativeqa.json baseline

# E-RECAP setup
bash scripts/run_lmeval_setup.sh data/LongBench/narrativeqa.json erecap
```

## 📋 Command-Line Arguments

### `run_lmeval.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--task_config` | Yes | - | Path to LongBench task JSON file |
| `--model_name` | No | `checkpoints/qwen2-7b-instruct` | Model name or path |
| `--pruner` | No | `None` | Path to pruning module (None for baseline) |
| `--output` | No | Auto-generated | Output JSON path for setup result |
| `--device` | No | `cuda` | Device to use |

### Example Output

```
============================================================
[LM-EVAL] E-RECAP Evaluation Setup (No Inference)
============================================================
Task config: data/LongBench/narrativeqa.json
Model: checkpoints/qwen2-7b-instruct
Pruning module: None (baseline)
Device: cuda
Output: results/lmeval_narrativeqa_baseline_setup.json
============================================================

[Step 1] Initializing model wrapper...
[E-RECAPModel] Initialized (no model loaded)
  Model: checkpoints/qwen2-7b-instruct
  Pruning module: None (baseline)
  Device: cuda
  Mode: Baseline
[OK] Model wrapper initialized

[Step 2] Loading LongBench task...
[LongBenchTask] Initialized: narrativeqa
  Data path: data/LongBench/narrativeqa.json
  Dataset size: 0 samples
[OK] Task loaded: 0 samples

[Step 3] Running setup evaluation (no inference)...
[LM-EVAL] LongBench task loaded: narrativeqa
[LM-EVAL] Data path: data/LongBench/narrativeqa.json
[LM-EVAL] Dataset size: 0 samples
[LM-EVAL] Model wrapper: E-RECAPModel
[LM-EVAL] Model config: checkpoints/qwen2-7b-instruct
[LM-EVAL] Mode: Baseline (no pruning)
[LM-EVAL] No inference executed in this setup phase.
[OK] Setup evaluation completed

[Step 4] Saving setup result...
[OK] Setup result saved to: results/lmeval_narrativeqa_baseline_setup.json

============================================================
[LM-EVAL] Setup completed successfully!
============================================================

Next steps:
1. Implement actual model loading in E-RECAPModel
2. Implement inference methods (generate_until, loglikelihood)
3. Run actual evaluation with lm-eval-harness
============================================================
```

## 🔧 Components

### 1. LongBenchTask (`longbench_task.py`)

Custom task class that adapts LongBench tasks to lm-eval-harness format.

**Key Methods:**
- `validation_docs()`: Iterator over validation documents
- `doc_to_text()`: Convert document to input text
- `doc_to_target()`: Extract target answer
- `evaluate()`: Setup-only evaluation (no inference)

### 2. E-RECAPModel (`erecap_model.py`)

Model wrapper that provides lm-eval-harness-compatible interface for E-RECAP models.

**Key Methods:**
- `generate_until()`: Generate text until stop sequences (setup phase: raises NotImplementedError)
- `loglikelihood()`: Compute log-likelihood (setup phase: raises NotImplementedError)
- `loglikelihood_rolling()`: Rolling window log-likelihood (setup phase: raises NotImplementedError)

**Attributes:**
- `model_name`: Model path or name
- `pruning_module`: Path to pruning module checkpoint (None for baseline)
- `is_erecap()`: Check if configured for E-RECAP

### 3. run_lmeval.py

Main execution script that:
1. Initializes E-RECAP model wrapper (no actual loading)
2. Loads LongBench task (data loading only)
3. Runs setup evaluation (no inference)
4. Saves setup result to JSON

## 📝 Expected Output Format

The setup result JSON contains:

```json
{
  "task": "narrativeqa",
  "task_path": "data/LongBench/narrativeqa.json",
  "dataset_size": 100,
  "model": "E-RECAPModel(name=checkpoints/qwen2-7b-instruct, mode=Baseline, pruner=None)",
  "status": "setup_completed",
  "message": "No inference executed in setup phase",
  "setup_info": {
    "model_name": "checkpoints/qwen2-7b-instruct",
    "pruning_module": null,
    "device": "cuda",
    "task_config": "data/LongBench/narrativeqa.json",
    "output_path": "results/lmeval_narrativeqa_baseline_setup.json"
  }
}
```

## 🔄 Integration with lm-eval-harness

In the actual evaluation phase, this framework will be integrated with lm-eval-harness:

1. **Task Registration**: Register `LongBenchTask` with lm-eval-harness task registry
2. **Model Integration**: Make `E-RECAPModel` compatible with lm-eval-harness LLM interface
3. **Evaluation Execution**: Run actual inference through lm-eval-harness CLI or API

## 🚧 TODO: Future Implementation

### Phase D (Actual Evaluation)

1. **Model Loading**:
   - Implement actual model loading in `E-RECAPModel.__init__()`
   - Load Qwen2-7B model and tokenizer
   - Load pruning module if provided

2. **Inference Methods**:
   - Implement `generate_until()` with E-RECAP pruning
   - Implement `loglikelihood()` with E-RECAP pruning
   - Handle stop sequences and generation parameters

3. **lm-eval-harness Integration**:
   - Register custom task with lm-eval-harness
   - Test with official lm-eval-harness CLI
   - Run full evaluation on all LongBench tasks

## 📚 Related Files

- **Phase 2C (LongBench)**: `src/evaluation/longbench/`
- **Unified API**: `src/evaluation/model_api.py`
- **Old Runner**: `src/evaluation/lmeval_runner.py` (subprocess-based, to be replaced)

## ✅ Verification Checklist

- [x] Directory structure created
- [x] LongBenchTask class implemented (no inference)
- [x] E-RECAPModel wrapper implemented (no inference)
- [x] Main script run_lmeval.py created (safe to run)
- [x] Task configuration YAML created
- [x] Shell script created with proper permissions
- [x] Documentation added

## 🎯 Next Steps

1. Test setup scripts with sample data
2. Verify output JSON format
3. Prepare for Phase D (actual inference implementation)
4. Integrate with lm-eval-harness task registry

---

**Phase**: D - Setup Complete ✅  
**Status**: Ready for actual inference implementation  
**Last Updated**: Phase D Setup

