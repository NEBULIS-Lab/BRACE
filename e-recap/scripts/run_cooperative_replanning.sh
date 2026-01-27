#!/bin/bash
# Run cooperative multi-agent planning with E-RECAP

set -e

# Default configuration
MODEL_PATH="${MODEL_PATH:-checkpoints/qwen2-7b-instruct}"
PRUNING_CKPT="${PRUNING_CKPT:-checkpoints/pruning_module.pt}"
KEEP_RATIO="${KEEP_RATIO:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TASK_TYPE="${TASK_TYPE:-iterative_replanning}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --pruning_ckpt)
            PRUNING_CKPT="$2"
            shift 2
            ;;
        --keep_ratio)
            KEEP_RATIO="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save_results)
            SAVE_RESULTS="--save_results"
            shift
            ;;
        --baseline)
            BASELINE="--baseline"
            shift
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$(dirname "$0")/.."

# Run the cooperative planning test
python3 src/multi_agent/run_cooperative_test.py \
    --model_path "$MODEL_PATH" \
    --pruning_ckpt "$PRUNING_CKPT" \
    --keep_ratio "$KEEP_RATIO" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --task_type "$TASK_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_RESULTS \
    $BASELINE \
    ${NUM_RUNS:+--num_runs $NUM_RUNS}

