"""
Compare Baseline vs E-RECAP for Cooperative Multi-Agent Planning.

This script runs both baseline (no pruning) and E-RECAP versions and compares
the results, including context growth, execution time, and compression ratios.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent.cooperative_planner import create_planner
from multi_agent.task_definitions import get_task_steps


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(baseline_file: str, erecap_file: str):
    """Compare baseline and E-RECAP results."""
    baseline = load_results(baseline_file)
    erecap = load_results(erecap_file)
    
    print("=" * 80)
    print("Baseline vs E-RECAP Comparison")
    print("=" * 80)
    print()
    
    # Check if multiple runs
    baseline_runs = baseline.get('num_runs', 1)
    erecap_runs = erecap.get('num_runs', 1)
    
    if baseline_runs > 1 or erecap_runs > 1:
        print(f"Multiple Runs: Baseline {baseline_runs} runs, E-RECAP {erecap_runs} runs")
        print()
        
        # Time comparison for multiple runs
        baseline_total = baseline.get('total_wall_clock_time', baseline.get('total_time', 0))
        erecap_total = erecap.get('total_wall_clock_time', erecap.get('total_time', 0))
        baseline_avg = baseline.get('average_time_per_run', baseline.get('total_time', 0) / baseline_runs)
        erecap_avg = erecap.get('average_time_per_run', erecap.get('total_time', 0) / erecap_runs)
        
        print("Execution Time (Multiple Runs):")
        print(f"  Baseline total (wall-clock): {baseline_total:.2f}s")
        print(f"  E-RECAP total (wall-clock):  {erecap_total:.2f}s")
        print(f"  Baseline average per run: {baseline_avg:.2f}s")
        print(f"  E-RECAP average per run:  {erecap_avg:.2f}s")
        speedup_total = baseline_total / erecap_total if erecap_total > 0 else 0
        speedup_avg = baseline_avg / erecap_avg if erecap_avg > 0 else 0
        print(f"  Speedup (total): {speedup_total:.2f}×")
        print(f"  Speedup (average): {speedup_avg:.2f}×")
        print()
        
        # Pruning overhead
        total_pruning = erecap.get('total_pruning_time', 0)
        print("Pruning Overhead:")
        print(f"  E-RECAP total pruning time: {total_pruning:.2f}s")
        print(f"  Pruning overhead: {total_pruning/erecap_total*100:.1f}%")
        print()
    else:
        # Single run comparison
        print("Execution Time:")
        print(f"  Baseline: {baseline['total_time']:.2f}s")
        print(f"  E-RECAP:  {erecap['total_time']:.2f}s")
        speedup = baseline['total_time'] / erecap['total_time'] if erecap['total_time'] > 0 else 0
        print(f"  Speedup:  {speedup:.2f}×")
        print()
        
        # Pruning overhead
        print("Pruning Overhead:")
        print(f"  E-RECAP pruning time: {erecap['total_pruning_time']:.2f}s")
        print(f"  Pruning overhead: {erecap['total_pruning_time']/erecap['total_time']*100:.1f}%")
        print()
    
    # Context growth comparison
    print("Context Growth (Final):")
    baseline_final = baseline['final_context_summary']['total_context_length']
    erecap_final = erecap['final_context_summary']['total_context_length']
    print(f"  Baseline final context: {baseline_final} chars")
    print(f"  E-RECAP final context:  {erecap_final} chars")
    reduction = (baseline_final - erecap_final) / baseline_final * 100 if baseline_final > 0 else 0
    print(f"  Context reduction: {reduction:.1f}%")
    print()
    
    # Step-by-step context comparison (use last run for multiple runs)
    baseline_history = baseline.get('last_run_planning_history', baseline.get('planning_history', []))
    erecap_history = erecap.get('last_run_planning_history', erecap.get('planning_history', []))
    
    if baseline_history and erecap_history:
        print("Step-by-Step Context Comparison (Last Run):")
        print(f"{'Step':<6} {'Baseline Before':<18} {'Baseline After':<17} {'E-RECAP Before':<18} {'E-RECAP After':<17} {'Compression':<12}")
        print("-" * 100)
        
        for i in range(min(len(baseline_history), len(erecap_history))):
            b_step = baseline_history[i]
            e_step = erecap_history[i]
            
            b_before = b_step['context_length_before']
            b_after = b_step['context_length_after']
            e_before = e_step['context_length_before']
            e_after = e_step['context_length_after']
            comp = e_step.get('compression_ratio', 1.0) * 100
            
            print(f"{i:<6} {b_before:<18} {b_after:<17} {e_before:<18} {e_after:<17} {comp:.1f}%")
        
        print()
        
        # Average compression
        erecap_comps = [s.get('compression_ratio', 1.0) for s in erecap_history]
        avg_comp = sum(erecap_comps) / len(erecap_comps) * 100 if erecap_comps else 0
        print(f"Average E-RECAP compression ratio: {avg_comp:.1f}%")
        print()
    
    # Show run statistics if multiple runs
    if baseline_runs > 1:
        print("Baseline Run Statistics:")
        runs = baseline.get('runs', [])
        if runs:
            times = [r['total_time'] for r in runs]
            print(f"  Min time: {min(times):.2f}s")
            print(f"  Max time: {max(times):.2f}s")
            print(f"  Avg time: {sum(times)/len(times):.2f}s")
        print()
    
    if erecap_runs > 1:
        print("E-RECAP Run Statistics:")
        runs = erecap.get('runs', [])
        if runs:
            times = [r['total_time'] for r in runs]
            print(f"  Min time: {min(times):.2f}s")
            print(f"  Max time: {max(times):.2f}s")
            print(f"  Avg time: {sum(times)/len(times):.2f}s")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs E-RECAP for cooperative multi-agent planning"
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        required=True,
        help="Path to baseline results JSON file",
    )
    parser.add_argument(
        "--erecap_file",
        type=str,
        required=True,
        help="Path to E-RECAP results JSON file",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline_file):
        print(f"Error: Baseline file not found: {args.baseline_file}")
        return 1
    
    if not os.path.exists(args.erecap_file):
        print(f"Error: E-RECAP file not found: {args.erecap_file}")
        return 1
    
    compare_results(args.baseline_file, args.erecap_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

