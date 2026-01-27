#!/usr/bin/env python3
"""
验证E-RECAP剪枝是否真正生效的脚本
检查：
1. Prefill阶段是否真的剪枝了token
2. 剪枝后的序列长度是否符合预期
3. 剪枝统计信息是否正确
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_erecap import (
    load_model_and_pruners,
    prefill_with_pruning,
    apply_token_pruning,
    KEEP07_CONFIG,
    KEEP09_CONFIG,
    KEEP08_CONFIG,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_pruning_stats(input_length: int, config_name: str = "keep07"):
    """Verify pruning statistics"""
    print(f"\n{'='*60}")
    print(f"Verifying pruning effect - Input length: {input_length}, Config: {config_name}")
    print(f"{'='*60}")
    
    # Select configuration
    if config_name == "keep09":
        config = KEEP09_CONFIG
    elif config_name == "keep08":
        config = KEEP08_CONFIG
    else:
        config = KEEP07_CONFIG
    
    # Load model
    model, tokenizer, pruners = load_model_and_pruners(prune_layers=config["prune_layers"])
    model.eval()
    
    # Create test input
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")["input_ids"][0]
    if base_ids.size(0) >= input_length:
        ids = base_ids[:input_length]
    else:
        repeat = (input_length + base_ids.size(0) - 1) // base_ids.size(0)
        ids = base_ids.repeat(repeat)[:input_length]
    
    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    print(f"\n[Input Information]")
    print(f"  Original sequence length: {input_ids.size(1)}")
    print(f"  Config: keep_ratio={config['keep_ratio']}, prune_layers={config['prune_layers']}")
    print(f"  Expected cumulative keep ratio: {config['cumulative_keep_ratio']:.4f}")
    
    # Execute prefill with pruning
    with torch.no_grad():
        logits, pruning_stats = prefill_with_pruning(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pruning_modules=pruners,
            keep_ratio=config["keep_ratio"],
            prune_layers=config["prune_layers"],
            min_head_tokens=config["min_head_tokens"],
            min_tail_ratio=config["min_tail_ratio"],
        )
    
    print(f"\n[Pruning Statistics]")
    print(f"  Number of pruning layers: {pruning_stats['total_pruning_steps']}")
    print(f"  Total tokens pruned: {pruning_stats['total_tokens_pruned']}")
    print(f"  Final sequence length: {pruning_stats['final_length']}")
    print(f"  Actual pruning ratio: {pruning_stats['total_tokens_pruned'] / input_length * 100:.2f}%")
    print(f"  Actual keep ratio: {pruning_stats['final_length'] / input_length * 100:.2f}%")
    
    # Check layer by layer
    print(f"\n[Layer-wise Pruning Details]")
    for layer_stat in pruning_stats['layer_stats']:
        layer_idx = layer_stat['layer']
        original = layer_stat['original_length']
        kept = layer_stat['tokens_kept']
        pruned = layer_stat['tokens_pruned']
        ratio = layer_stat['pruning_ratio']
        print(f"  Layer {layer_idx:2d}: {original:5d} -> {kept:5d} tokens "
              f"(pruned {pruned:5d}, {ratio*100:5.2f}%)")
    
    # Verify if pruning actually took effect
    print(f"\n[Verification Results]")
    if pruning_stats['total_pruning_steps'] == 0:
        print("  ERROR: No pruning operations were executed!")
        return False
    
    if pruning_stats['final_length'] >= input_length:
        print("  ERROR: Final sequence length did not decrease, pruning may not have taken effect!")
        return False
    
    expected_min_keep = int(input_length * config['cumulative_keep_ratio'] * 0.8)  # Allow 20% error
    expected_max_keep = int(input_length * config['cumulative_keep_ratio'] * 1.2)  # Allow 20% error
    
    if pruning_stats['final_length'] < expected_min_keep:
        print(f"  WARNING: Final sequence length ({pruning_stats['final_length']}) is less than expected minimum ({expected_min_keep})")
        print(f"     Pruning may be too aggressive")
    elif pruning_stats['final_length'] > expected_max_keep:
        print(f"  WARNING: Final sequence length ({pruning_stats['final_length']}) is greater than expected maximum ({expected_max_keep})")
        print(f"     Pruning may not be aggressive enough")
    else:
        print(f"  PASS: Pruning effect meets expectations")
        print(f"     Expected range: {expected_min_keep} - {expected_max_keep}")
        print(f"     Actual value: {pruning_stats['final_length']}")
    
    # Check logits shape
    print(f"\n[Output Verification]")
    print(f"  Logits shape: {logits.shape}")
    expected_logits_shape = (1, pruning_stats['final_length'], model.config.vocab_size)
    if logits.shape == expected_logits_shape:
        print(f"  PASS: Logits shape is correct: {logits.shape}")
    else:
        print(f"  ERROR: Logits shape is incorrect: expected {expected_logits_shape}, got {logits.shape}")
        return False
    
    return True


def verify_layer_wise_pruning():
    """Verify that layer-wise pruning works as expected"""
    print(f"\n{'='*60}")
    print(f"Verifying Layer-wise Pruning Mechanism")
    print(f"{'='*60}")
    
    model, tokenizer, pruners = load_model_and_pruners()
    model.eval()
    
    input_length = 4096
    base_ids = tokenizer("Hello, this is a test.", return_tensors="pt")["input_ids"][0]
    repeat = (input_length + base_ids.size(0) - 1) // base_ids.size(0)
    ids = base_ids.repeat(repeat)[:input_length]
    input_ids = ids.unsqueeze(0).to(device)
    
    # Manually execute first few layers to check pruning effect
    hidden_states = model.model.embed_tokens(input_ids)
    print(f"\n[Initial] Sequence length: {hidden_states.size(1)}")
    
    config = KEEP07_CONFIG
    keep_ratio = config["keep_ratio"]
    prune_layers = config["prune_layers"]
    
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx >= max(prune_layers) + 1:  # Only check first few pruning layers
            break
            
        position_ids = torch.arange(
            0, hidden_states.size(1), dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0)
        
        outputs = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]
        
        if layer_idx in prune_layers:
            pruner = pruners[str(layer_idx)]
            hidden_states, index_tensor, stats = apply_token_pruning(
                hidden_states,
                pruner,
                keep_ratio,
                config["min_head_tokens"],
                config["min_tail_ratio"],
            )
            print(f"  Layer {layer_idx:2d}: {stats['original_length']:5d} -> {stats['tokens_kept']:5d} "
                  f"(pruned {stats['tokens_pruned']:5d}, {stats['pruning_ratio']*100:5.2f}%)")
        else:
            print(f"  Layer {layer_idx:2d}: {hidden_states.size(1):5d} (not pruned)")
    
    print(f"\n[Final] Sequence length: {hidden_states.size(1)}")
    return True


def main():
    print("="*60)
    print("E-RECAP Pruning Verification Script")
    print("="*60)
    
    # Test different configurations
    test_configs = ["keep09", "keep08", "keep07"]
    test_lengths = [1024, 4096, 8192]
    
    all_passed = True
    
    for config_name in test_configs:
        for length in test_lengths:
            try:
                passed = verify_pruning_stats(length, config_name)
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"\nERROR: Test failed: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
    
    # Verify layer-wise pruning
    try:
        verify_layer_wise_pruning()
    except Exception as e:
        print(f"\nERROR: Layer-wise pruning verification failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("PASS: All verifications passed! Pruning is indeed taking effect.")
    else:
        print("ERROR: Some verifications failed, please check the implementation.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

