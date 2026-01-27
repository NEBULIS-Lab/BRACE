# 多 GPU 测试（不带 E-RECAP，只测试显存上限）

import argparse
import subprocess
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "checkpoints/qwen2-7b-instruct"

# ======================================================
# GPU memory query
# ======================================================
def get_gpu_memory():
    """
    Returns a list of (used_MB, total_MB) per GPU.
    """
    query = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.strip().split("\n")

    result = []
    for line in query:
        used, total = line.split(",")
        result.append((int(used), int(total)))
    return result


# ======================================================
# Auto-select GPU with most free memory
# ======================================================
def pick_best_gpu():
    mem = get_gpu_memory()
    free = [(idx, total - used, used, total) for idx, (used, total) in enumerate(mem)]
    best = max(free, key=lambda x: x[1])  # x[1] = free memory
    gpu_id, free_mb, used_mb, total_mb = best
    print(f"[GPU Select] Chosen GPU {gpu_id} | free={free_mb}MB used={used_mb}MB total={total_mb}MB")
    return gpu_id


# ======================================================
# Build a dummy long input
# ======================================================
def build_dummy_input(tokenizer, length, device):
    base = tokenizer("Hello world.", return_tensors="pt")["input_ids"][0]
    if base.size(0) >= length:
        ids = base[:length]
    else:
        repeat = (length + base.size(0) - 1) // base.size(0)
        ids = base.repeat(repeat)[:length]

    ids = ids.unsqueeze(0).to(device)
    mask = torch.ones_like(ids).to(device)
    return ids, mask


# ======================================================
# Measure baseline forward memory & time
# ======================================================
def measure_forward(model, ids, mask):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    return time.perf_counter() - start


# ======================================================
# Main multi-GPU test
# ======================================================
def run_test(lengths):
    # pick GPU with most free memory
    gpu_id = pick_best_gpu()
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[Loading Model] onto GPU {gpu_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()

    print("[Model loaded]")

    for L in lengths:
        print("\n============================================")
        print(f"Testing sequence length: {L}")
        print("============================================")

        used, total = get_gpu_memory()[gpu_id]
        print(f"[Before Build Input] GPU{gpu_id}: used={used}MB / total={total}MB")

        ids, mask = build_dummy_input(tokenizer, L, device)

        used, total = get_gpu_memory()[gpu_id]
        print(f"[After Build Input] GPU{gpu_id}: used={used}MB / total={total}MB")

        try:
            t = measure_forward(model, ids, mask)
            used_after, total_after = get_gpu_memory()[gpu_id]

            print(f"[Forward OK] time={t:.4f}s")
            print(f"[After Forward] GPU{gpu_id}: used={used_after}MB / total={total_after}MB")

        except torch.cuda.OutOfMemoryError:
            used_after, total_after = get_gpu_memory()[gpu_id]
            print(f"[OOM] GPU{gpu_id}: used={used_after}MB / total={total_after}MB")
            print("[Skipping this length]")

        finally:
            del ids, mask
            torch.cuda.empty_cache()


# ======================================================
# CLI
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args.lengths)
