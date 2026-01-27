"""
Latency curve plotting script for E-RECAP
Generates latency, speedup, and FLOPs reduction curves
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def load_json(path):
    """
    Load JSON file and extract latency data.
    Supports both old format ({length: latency}) and new format (with metadata/results).
    
    Returns:
        dict: {length: latency} mapping
        dict: Metadata if available (new format), None otherwise
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}, None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Check if it's new format (has 'results' or 'metadata' key)
    if "results" in data:
        # New format: extract latency from results
        latency_data = {}
        for length_str, result in data["results"].items():
            if isinstance(result, dict) and "baseline" in result:
                # Combined format: extract baseline latency
                latency_data[int(length_str)] = result["baseline"]["latency_seconds"]
            elif isinstance(result, dict) and "latency_seconds" in result:
                # Separate baseline/erecap format
                latency_data[int(length_str)] = result["latency_seconds"]
            else:
                # Fallback: assume it's a number
                latency_data[int(length_str)] = float(result)
        metadata = data.get("metadata", None)
        return latency_data, metadata
    elif "metadata" in data:
        # New format but only metadata (shouldn't happen, but handle it)
        return {}, data.get("metadata", None)
    else:
        # Old format: simple {length: latency}
        return {int(k): float(v) for k, v in data.items()}, None


def plot_latency(baseline, erecap, out_path):
    """
    Plot prefill latency vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        erecap: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    base_vals = [baseline[L] for L in lengths]
    erecap_vals = [erecap[L] for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_vals, marker='o', linewidth=2, label="Baseline", markersize=8)
    plt.plot(lengths, erecap_vals, marker='s', linewidth=2, label="E-RECAP", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Prefill Latency (seconds)", fontsize=12)
    plt.title("Prefill Latency vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def plot_speedup(baseline, erecap, out_path):
    """
    Plot speedup vs sequence length
    
    Args:
        baseline: Dict mapping sequence length to latency (seconds)
        erecap: Dict mapping sequence length to latency (seconds)
        out_path: Output file path
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    speedups = [baseline[L] / erecap[L] if erecap[L] > 0 else 0 for L in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, speedups, marker='o', linewidth=2, label="Speedup", 
             markersize=8, color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Speedup (Baseline / E-RECAP)", fontsize=12)
    plt.title("Speedup vs Sequence Length", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average speedup: {np.mean(speedups):.2f}x")


def estimate_flops(length, keep_ratio=0.7, hidden_size=3584, num_layers=28, num_heads=32):
    """
    Estimate FLOPs for Transformer forward pass
    
    Args:
        length: Sequence length
        keep_ratio: Token keep ratio (for E-RECAP)
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Attention head dimension
    
    Returns:
        Estimated FLOPs (normalized)
    """
    head_dim = hidden_size // num_heads
    
    # Attention FLOPs: 4 * L^2 * d (QK^T, softmax, AV)
    # MLP FLOPs: 8 * L * d^2 (two linear layers, expansion ratio ~4)
    # Per layer FLOPs
    attn_flops = 4 * length * length * hidden_size
    mlp_flops = 8 * length * hidden_size * hidden_size
    
    # Total FLOPs for all layers
    total_flops = num_layers * (attn_flops + mlp_flops)
    
    return total_flops


def plot_flops(baseline, erecap, out_path, keep_ratio=0.7):
    """
    Plot estimated FLOPs reduction
    
    Args:
        baseline: Dict mapping sequence length to latency (for reference)
        erecap: Dict mapping sequence length to latency (for reference)
        out_path: Output file path
        keep_ratio: Average token keep ratio for E-RECAP
    """
    # Get common lengths
    lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
    if not lengths:
        print("[Error] No common sequence lengths found")
        return
    
    # Estimate FLOPs
    base_flops = [estimate_flops(L) for L in lengths]
    
    # E-RECAP FLOPs: tokens are pruned progressively, use average keep ratio
    # For simplicity, use keep_ratio for all layers (in reality it's progressive)
    erecap_flops = [estimate_flops(int(L * keep_ratio)) for L in lengths]
    
    # Normalize to first value for better visualization
    if base_flops[0] > 0:
        base_flops_norm = [f / base_flops[0] for f in base_flops]
        erecap_flops_norm = [f / base_flops[0] for f in erecap_flops]
    else:
        base_flops_norm = base_flops
        erecap_flops_norm = erecap_flops
    
    reduction = [(1 - erecap_flops[i] / base_flops[i]) * 100 
                 for i in range(len(lengths))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, base_flops_norm, marker='o', linewidth=2, 
             label="Baseline FLOPs", markersize=8)
    plt.plot(lengths, erecap_flops_norm, marker='s', linewidth=2, 
             label=f"E-RECAP FLOPs (keep_ratio={keep_ratio})", markersize=8)
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Relative FLOPs (normalized)", fontsize=12)
    plt.title("Estimated FLOPs Reduction", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")
    print(f"[Info] Average FLOPs reduction: {np.mean(reduction):.1f}%")


def load_combined_json(path):
    """
    Load combined JSON file (new format with both baseline and E-RECAP).
    
    Returns:
        tuple: (baseline_dict, erecap_dict, metadata)
    """
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}, {}, None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if "results" not in data:
        print(f"[Warning] Invalid format in {path}, expected 'results' key")
        return {}, {}, None
    
    baseline = {}
    erecap = {}
    
    for length_str, result in data["results"].items():
        length = int(length_str)
        if isinstance(result, dict):
            if "baseline" in result:
                baseline[length] = result["baseline"]["latency_seconds"]
            if "erecap" in result:
                erecap[length] = result["erecap"]["latency_seconds"]
    
    metadata = data.get("metadata", None)
    return baseline, erecap, metadata


def plot_multi_config_latency(configs_data, out_path):
    """
    Plot latency comparison for multiple configurations in one figure.
    
    Args:
        configs_data: Dict of {config_name: (baseline_dict, erecap_dict)}
        out_path: Output file path
    """
    plt.figure(figsize=(12, 7))
    
    # Plot baseline (only once, as it's the same for all configs)
    if configs_data:
        first_config = list(configs_data.keys())[0]
        baseline, _ = configs_data[first_config]
        lengths = sorted(baseline.keys())
        base_vals = [baseline[L] for L in lengths]
        plt.plot(lengths, base_vals, marker='o', linewidth=2.5, 
                label="Baseline", markersize=10, color='black', linestyle='--')
    
    # Plot E-RECAP for each configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['s', '^', 'D']  # Square, Triangle, Diamond
    
    for idx, (config_name, (baseline, erecap)) in enumerate(configs_data.items()):
        lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
        erecap_vals = [erecap[L] for L in lengths]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(lengths, erecap_vals, marker=marker, linewidth=2, 
                label=f"E-RECAP ({config_name})", markersize=8, color=color)
    
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Prefill Latency (seconds)", fontsize=12)
    plt.title("Prefill Latency Comparison: Multiple Configurations", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def plot_multi_config_speedup(configs_data, out_path):
    """
    Plot speedup comparison for multiple configurations in one figure.
    
    Args:
        configs_data: Dict of {config_name: (baseline_dict, erecap_dict)}
        out_path: Output file path
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['s', '^', 'D']  # Square, Triangle, Diamond
    
    for idx, (config_name, (baseline, erecap)) in enumerate(configs_data.items()):
        lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
        speedups = [baseline[L] / erecap[L] if erecap[L] > 0 else 0 for L in lengths]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(lengths, speedups, marker=marker, linewidth=2, 
                label=f"{config_name} (avg: {np.mean(speedups):.2f}x)", 
                markersize=8, color=color)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Speedup (Baseline / E-RECAP)", fontsize=12)
    plt.title("Speedup Comparison: Multiple Configurations", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate latency curves for E-RECAP evaluation"
    )
    parser.add_argument(
        "--baseline", 
        type=str, 
        default=None,
        help="Path to baseline latency JSON file (old format or separate file)"
    )
    parser.add_argument(
        "--erecap", 
        type=str, 
        default=None,
        help="Path to E-RECAP latency JSON file (old format or separate file)"
    )
    parser.add_argument(
        "--combined",
        type=str,
        default=None,
        help="Path to combined results JSON file (new format with both baseline and E-RECAP)"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="results/fig",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=None,
        help="Token keep ratio for FLOPs estimation (auto-detect from metadata if not set)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g., 'singlegpu_' or 'multigpu_')"
    )
    parser.add_argument(
        "--multi_config",
        action="store_true",
        help="Plot multiple configurations in one figure (requires --combined files)"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        nargs="+",
        default=None,
        help="List of combined JSON files for multi-config plotting"
    )
    
    args = parser.parse_args()
    
    # Load data - support both old and new formats
    baseline = {}
    erecap = {}
    metadata = None
    
    if args.combined:
        # New format: load from combined file
        print(f"[Loading] Combined results: {args.combined}")
        baseline, erecap, metadata = load_combined_json(args.combined)
        if metadata and args.keep_ratio is None:
            # Auto-detect keep_ratio from metadata
            args.keep_ratio = metadata.get("pruning_config", {}).get("keep_ratio", 0.7)
    else:
        # Old format: load separate files
        if args.baseline is None:
            args.baseline = "results/latency_baseline.json"
        if args.erecap is None:
            args.erecap = "results/latency_erecap.json"
        
        print(f"[Loading] Baseline: {args.baseline}")
        baseline, baseline_meta = load_json(args.baseline)
        
        print(f"[Loading] E-RECAP: {args.erecap}")
        erecap, erecap_meta = load_json(args.erecap)
        
        metadata = baseline_meta or erecap_meta
        if metadata and args.keep_ratio is None:
            args.keep_ratio = metadata.get("pruning_config", {}).get("keep_ratio", 0.7)
    
    if not baseline:
        print("[Error] Baseline data is empty")
        return
    
    if not erecap:
        print("[Error] E-RECAP data is empty")
        return
    
    if args.keep_ratio is None:
        args.keep_ratio = 0.7  # Default fallback
    
    print(f"[Info] Found {len(baseline)} baseline points, {len(erecap)} E-RECAP points")
    if metadata:
        config_name = metadata.get("config_name", "unknown")
        print(f"[Info] Configuration: {config_name}")
    
    # Generate plots with optional prefix
    prefix = f"{args.prefix}_" if args.prefix else ""
    plot_latency(baseline, erecap, os.path.join(args.out_dir, f"{prefix}latency_curve.png"))
    plot_speedup(baseline, erecap, os.path.join(args.out_dir, f"{prefix}speedup_curve.png"))
    plot_flops(baseline, erecap, os.path.join(args.out_dir, f"{prefix}flops_curve.png"), 
               keep_ratio=args.keep_ratio)
    
    print("[OK] All plots generated successfully!")


def plot_multi_config_flops(configs_data, keep_ratios, out_path):
    """
    Plot FLOPs comparison for multiple configurations in one figure.
    
    Args:
        configs_data: Dict of {config_name: (baseline_dict, erecap_dict)}
        keep_ratios: Dict of {config_name: keep_ratio}
        out_path: Output file path
    """
    plt.figure(figsize=(12, 7))
    
    # Get common lengths from first config
    if not configs_data:
        return
    first_config = list(configs_data.keys())[0]
    baseline, _ = configs_data[first_config]
    lengths = sorted(baseline.keys())
    
    # Plot baseline FLOPs (only once)
    base_flops = [estimate_flops(L) for L in lengths]
    if base_flops[0] > 0:
        base_flops_norm = [f / base_flops[0] for f in base_flops]
    else:
        base_flops_norm = base_flops
    plt.plot(lengths, base_flops_norm, marker='o', linewidth=2.5, 
            label="Baseline FLOPs", markersize=10, color='black', linestyle='--')
    
    # Plot E-RECAP FLOPs for each configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['s', '^', 'D']  # Square, Triangle, Diamond
    
    for idx, (config_name, (baseline, erecap)) in enumerate(configs_data.items()):
        keep_ratio = keep_ratios.get(config_name, 0.7)
        erecap_flops = [estimate_flops(int(L * keep_ratio)) for L in lengths]
        if base_flops[0] > 0:
            erecap_flops_norm = [f / base_flops[0] for f in erecap_flops]
        else:
            erecap_flops_norm = erecap_flops
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(lengths, erecap_flops_norm, marker=marker, linewidth=2, 
                label=f"E-RECAP {config_name} (keep_ratio={keep_ratio})", 
                markersize=8, color=color)
    
    plt.xlabel("Sequence Length (tokens)", fontsize=12)
    plt.ylabel("Relative FLOPs (normalized)", fontsize=12)
    plt.title("FLOPs Comparison: Multiple Configurations", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved {out_path}")


def plot_comprehensive_singlegpu(out_dir="results/fig"):
    """
    Plot comprehensive single-GPU comparison with multiple subplots.
    Shows latency, speedup, and FLOPs reduction in one figure.
    """
    config_files = [
        ("keep09", "results/latency_results_keep09.json"),
        ("keep08", "results/latency_results_keep08.json"),
        ("keep07", "results/latency_results_keep07.json"),
    ]
    
    configs_data = {}
    keep_ratios = {}
    metadata_dict = {}
    
    for config_name, config_file in config_files:
        if os.path.exists(config_file):
            baseline, erecap, metadata = load_combined_json(config_file)
            if baseline and erecap:
                final_name = metadata.get("config_name", config_name) if metadata else config_name
                configs_data[final_name] = (baseline, erecap)
                if metadata:
                    keep_ratios[final_name] = metadata.get("pruning_config", {}).get("keep_ratio", 0.7)
                    metadata_dict[final_name] = metadata
                print(f"[Loaded] {final_name}: {len(baseline)} data points")
    
    if not configs_data:
        print("[Warning] No single-GPU configuration files found")
        return
    
    # Get common lengths
    first_config = list(configs_data.keys())[0]
    baseline, _ = configs_data[first_config]
    lengths = sorted(baseline.keys())
    
    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Single-GPU E-RECAP Performance: Comprehensive Comparison", fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['s', '^', 'D']  # Square, Triangle, Diamond
    
    # Subplot 1: Latency Comparison (Line plot)
    ax1 = axes[0, 0]
    base_vals = [baseline[L] for L in lengths]
    ax1.plot(lengths, base_vals, marker='o', linewidth=2.5, label="Baseline", 
            markersize=10, color='black', linestyle='--', zorder=1)
    
    for idx, (config_name, (baseline_dict, erecap_dict)) in enumerate(configs_data.items()):
        erecap_vals = [erecap_dict[L] for L in lengths]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax1.plot(lengths, erecap_vals, marker=marker, linewidth=2, 
                label=f"E-RECAP ({config_name})", markersize=8, color=color, zorder=2)
    
    ax1.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax1.set_ylabel("Prefill Latency (seconds)", fontsize=11)
    ax1.set_title("(a) Latency Comparison", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # Subplot 2: Speedup Comparison (Line plot)
    ax2 = axes[0, 1]
    for idx, (config_name, (baseline_dict, erecap_dict)) in enumerate(configs_data.items()):
        speedups = [baseline_dict[L] / erecap_dict[L] if erecap_dict[L] > 0 else 0 for L in lengths]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        avg_speedup = np.mean(speedups)
        ax2.plot(lengths, speedups, marker=marker, linewidth=2, 
                label=f"{config_name} (avg: {avg_speedup:.2f}x)", 
                markersize=8, color=color, zorder=2)
    
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (1x)', zorder=1)
    ax2.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax2.set_ylabel("Speedup (Baseline / E-RECAP)", fontsize=11)
    ax2.set_title("(b) Speedup vs Sequence Length", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Subplot 3: Average Speedup by Configuration (Bar chart)
    ax3 = axes[1, 0]
    config_names = []
    avg_speedups = []
    bar_colors = []
    
    for idx, (config_name, (baseline_dict, erecap_dict)) in enumerate(configs_data.items()):
        speedups = [baseline_dict[L] / erecap_dict[L] if erecap_dict[L] > 0 else 0 for L in lengths]
        avg_speedup = np.mean(speedups)
        config_names.append(config_name)
        avg_speedups.append(avg_speedup)
        bar_colors.append(colors[idx % len(colors)])
    
    bars = ax3.bar(config_names, avg_speedups, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (1x)')
    ax3.set_ylabel("Average Speedup", fontsize=11)
    ax3.set_title("(c) Average Speedup by Configuration", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=9)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, avg_speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 4: FLOPs Reduction (Bar chart)
    ax4 = axes[1, 1]
    config_names_flops = []
    flops_reductions = []
    bar_colors_flops = []
    
    for idx, (config_name, (baseline_dict, erecap_dict)) in enumerate(configs_data.items()):
        keep_ratio = keep_ratios.get(config_name, 0.7)
        # Calculate average FLOPs reduction across all lengths
        reductions = []
        for L in lengths:
            base_flops = estimate_flops(L)
            erecap_flops = estimate_flops(int(L * keep_ratio))
            reduction = (1 - erecap_flops / base_flops) * 100 if base_flops > 0 else 0
            reductions.append(reduction)
        avg_reduction = np.mean(reductions)
        
        config_names_flops.append(config_name)
        flops_reductions.append(avg_reduction)
        bar_colors_flops.append(colors[idx % len(colors)])
    
    bars_flops = ax4.bar(config_names_flops, flops_reductions, color=bar_colors_flops, 
                         alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel("FLOPs Reduction (%)", fontsize=11)
    ax4.set_title("(d) Average FLOPs Reduction by Configuration", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, reduction in zip(bars_flops, flops_reductions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    out_path = os.path.join(out_dir, "singlegpu_comprehensive.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved comprehensive single-GPU plot: {out_path}")


def plot_comprehensive_multigpu(out_dir="results/fig"):
    """
    Plot comprehensive multi-GPU comparison with multiple subplots.
    """
    baseline_file = "results/latency_baseline_multigpu.json"
    erecap_file = "results/latency_erecap_multigpu.json"
    
    if not (os.path.exists(baseline_file) and os.path.exists(erecap_file)):
        print("[Warning] Multi-GPU result files not found")
        return False
    
    baseline, _ = load_json(baseline_file)
    erecap, _ = load_json(erecap_file)
    
    if not baseline or not erecap:
        print("[Warning] Multi-GPU data is empty")
        return False
    
    lengths = sorted(set(baseline.keys()) & set(erecap.keys()))
    if not lengths:
        print("[Warning] No common sequence lengths in multi-GPU data")
        return False
    
    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-GPU E-RECAP Performance: Comprehensive Analysis", fontsize=16, fontweight='bold', y=0.995)
    
    base_vals = [baseline[L] for L in lengths]
    erecap_vals = [erecap[L] for L in lengths]
    speedups = [baseline[L] / erecap[L] if erecap[L] > 0 else 0 for L in lengths]
    avg_speedup = np.mean(speedups)
    
    # Subplot 1: Latency Comparison (Line plot)
    ax1 = axes[0, 0]
    ax1.plot(lengths, base_vals, marker='o', linewidth=2.5, label="Baseline", 
            markersize=10, color='black', linestyle='--', zorder=1)
    ax1.plot(lengths, erecap_vals, marker='s', linewidth=2, label="E-RECAP", 
            markersize=10, color='#2ca02c', zorder=2)
    ax1.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax1.set_ylabel("Prefill Latency (seconds)", fontsize=11)
    ax1.set_title("(a) Latency Comparison", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Subplot 2: Speedup vs Sequence Length (Line plot)
    ax2 = axes[0, 1]
    ax2.plot(lengths, speedups, marker='o', linewidth=2.5, label=f"Speedup (avg: {avg_speedup:.2f}x)", 
            markersize=10, color='#1f77b4', zorder=2)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (1x)', zorder=1)
    ax2.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax2.set_ylabel("Speedup (Baseline / E-RECAP)", fontsize=11)
    ax2.set_title("(b) Speedup vs Sequence Length", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    # Subplot 3: Speedup by Length (Bar chart)
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(lengths)), speedups, color='#1f77b4', alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (1x)')
    ax3.set_xticks(range(len(lengths)))
    ax3.set_xticklabels([str(L) for L in lengths], rotation=45, ha='right')
    ax3.set_ylabel("Speedup", fontsize=11)
    ax3.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax3.set_title("(c) Speedup by Sequence Length", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=9)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot 4: Latency Reduction Percentage (Bar chart)
    ax4 = axes[1, 1]
    latency_reductions = [(1 - erecap[L] / baseline[L]) * 100 if baseline[L] > 0 else 0 for L in lengths]
    avg_reduction = np.mean(latency_reductions)
    
    bars_reduction = ax4.bar(range(len(lengths)), latency_reductions, color='#ff7f0e', alpha=0.7,
                            edgecolor='black', linewidth=1.5)
    ax4.axhline(y=avg_reduction, color='r', linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'Average ({avg_reduction:.1f}%)')
    ax4.set_xticks(range(len(lengths)))
    ax4.set_xticklabels([str(L) for L in lengths], rotation=45, ha='right')
    ax4.set_ylabel("Latency Reduction (%)", fontsize=11)
    ax4.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax4.set_title("(d) Latency Reduction by Sequence Length", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    
    # Add value labels on bars
    for bar, reduction in zip(bars_reduction, latency_reductions):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    out_path = os.path.join(out_dir, "multigpu_comprehensive.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved comprehensive multi-GPU plot: {out_path}")
    return True


def plot_all_singlegpu_configs(out_dir="results/fig"):
    """
    Plot all single-GPU configurations (keep09, keep08, keep07) in one comprehensive figure.
    """
    plot_comprehensive_singlegpu(out_dir)


def plot_multigpu_results(out_dir="results/fig"):
    """
    Plot multi-GPU results in comprehensive figure.
    """
    return plot_comprehensive_multigpu(out_dir)


if __name__ == "__main__":
    import sys
    
    # Check if called with special mode for multi-config plotting
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-all-singlegpu":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/fig"
        plot_all_singlegpu_configs(out_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--plot-multigpu":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/fig"
        plot_multigpu_results(out_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--plot-comprehensive":
        # Plot both single-GPU and multi-GPU comprehensive figures
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/fig"
        plot_all_singlegpu_configs(out_dir)
        plot_multigpu_results(out_dir)
    else:
        main()

