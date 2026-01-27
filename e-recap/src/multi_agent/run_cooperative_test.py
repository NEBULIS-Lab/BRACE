"""
Test script for Cooperative Multi-Agent Planning with E-RECAP.

This script demonstrates the cooperative multi-agent planning setting where
multiple agents operate sequentially with E-RECAP token pruning applied to
the shared context buffer before each agent invocation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent.cooperative_planner import create_planner
from multi_agent.task_definitions import get_task_steps


def main():
    parser = argparse.ArgumentParser(
        description="Run cooperative multi-agent planning with E-RECAP"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/qwen2-7b-instruct",
        help="Path to the language model",
    )
    parser.add_argument(
        "--pruning_ckpt",
        type=str,
        default="checkpoints/pruning_module.pt",
        help="Path to the pruning module checkpoint",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.7,
        help="Fraction of tokens to keep per layer during pruning",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per agent",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="iterative_replanning",
        choices=["iterative_replanning", "embodied", "cooperative"],
        help="Type of task. Options: 'iterative_replanning' (15 steps, default, better for showcasing E-RECAP), 'embodied' (6 steps), 'cooperative' (8 steps, legacy)",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Design and plan a distributed AI training platform that supports multiple LLM training jobs with efficient resource allocation and monitoring.",
        help="Initial task description",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline (no pruning) for comparison",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of times to run the same task (default: 1, use 10 for longer evaluation)",
    )
    
    args = parser.parse_args()
    
    mode = "Baseline (no pruning)" if args.baseline else "E-RECAP"
    print("=" * 80)
    print(f"Cooperative Multi-Agent Planning - {mode}")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    if not args.baseline:
        print(f"Pruning checkpoint: {args.pruning_ckpt}")
        print(f"Keep ratio: {args.keep_ratio}")
    print(f"Task type: {args.task_type}")
    print(f"Task description: {args.task_description[:100]}...")
    print("=" * 80)
    print()
    
    # Create planner
    if args.baseline:
        print("Loading model (baseline mode, no pruning)...")
        try:
            from multi_agent.cooperative_planner import CooperativeMultiAgentPlanner
            from inference_erecap import load_model_and_pruners
            import torch.nn as nn
            
            model, tokenizer, _ = load_model_and_pruners()
            planner = CooperativeMultiAgentPlanner(
                model=model,
                tokenizer=tokenizer,
                pruning_modules=nn.ModuleDict(),  # Empty ModuleDict for baseline
                keep_ratio=args.keep_ratio,
                max_new_tokens=args.max_new_tokens,
            )
            print("✓ Model loaded (baseline mode)")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Loading model and pruning modules...")
        try:
            planner = create_planner(
                model_path=args.model_path,
                pruning_ckpt=args.pruning_ckpt,
                keep_ratio=args.keep_ratio,
                max_new_tokens=args.max_new_tokens,
            )
            print("✓ Model and pruning modules loaded")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return 1
    
    # Get task steps
    task_steps = get_task_steps(args.task_type)
    print(f"✓ Loaded {len(task_steps)} task steps")
    if args.num_runs > 1:
        print(f"✓ Will run {args.num_runs} times")
    print()
    
    # Run planning cycle (possibly multiple times)
    all_runs_results = []
    total_start_time = time.time()
    
    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print("=" * 80)
            print(f"Run {run_idx + 1}/{args.num_runs}")
            print("=" * 80)
            print()
        
        print("Starting cooperative planning cycle...")
        print("-" * 80)
        
        try:
            results = planner.run_planning_cycle(
                task_description=args.task_description,
                task_steps=task_steps,
                task_type=args.task_type,
                use_pruning=not args.baseline,
            )
            
            all_runs_results.append(results)
            
            print("-" * 80)
            print()
            print(f"Planning cycle {run_idx + 1} completed!")
            print()
            
            # Print summary for this run
            summary = planner.get_planning_summary()
            print(f"Run {run_idx + 1} Summary:")
            print(f"  Total steps: {summary['num_steps']}")
            print(f"  Total time: {summary['total_time']:.2f}s")
            print(f"  Pruning time: {summary['total_pruning_time']:.2f}s")
            print(f"  Inference time: {summary['total_inference_time']:.2f}s")
            print()
            
            if args.num_runs == 1:
                print("Context growth:")
                for growth in summary["context_growth"]:
                    print(f"  Step {growth['step']}: "
                          f"{growth['length_before']} -> {growth['length_after']} chars "
                          f"({growth['compression_ratio']:.2%})")
                print()
        
        except Exception as e:
            print(f"✗ Error during planning cycle {run_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    total_time = time.time() - total_start_time
    
    # Aggregate results across all runs
    if args.num_runs > 1:
        print("=" * 80)
        print("Aggregated Results Across All Runs")
        print("=" * 80)
        print()
        
        total_time_all = sum(r["total_time"] for r in all_runs_results)
        total_pruning_time_all = sum(r["total_pruning_time"] for r in all_runs_results)
        total_inference_time_all = sum(r["total_inference_time"] for r in all_runs_results)
        
        print(f"Total runs: {args.num_runs}")
        print(f"Total wall-clock time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Total execution time (sum): {total_time_all:.2f}s")
        print(f"Average time per run: {total_time_all / args.num_runs:.2f}s")
        print(f"Total pruning time: {total_pruning_time_all:.2f}s")
        print(f"Total inference time: {total_inference_time_all:.2f}s")
        print()
        
        # Get final context from last run
        final_summary = all_runs_results[-1]["final_context_summary"]
        print(f"Final context (last run): {final_summary['total_context_length']} chars")
        print()
    
    # Save results if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        mode_suffix = "baseline" if args.baseline else f"{args.keep_ratio}"
        if args.num_runs > 1:
            output_file = os.path.join(
                args.output_dir,
                f"cooperative_planning_{args.task_type}_{mode_suffix}_{args.num_runs}runs.json"
            )
        else:
            output_file = os.path.join(
                args.output_dir,
                f"cooperative_planning_{args.task_type}_{mode_suffix}.json"
            )
        
        # Convert results to JSON-serializable format
        if args.num_runs == 1:
            # Single run format
            results = all_runs_results[0]
            json_results = {
                "mode": "baseline" if args.baseline else "erecap",
                "num_runs": 1,
                "task_description": results["task_description"],
                "num_agents": results["num_agents"],
                "total_time": results["total_time"],
                "total_pruning_time": results["total_pruning_time"],
                "total_inference_time": results["total_inference_time"],
                "final_context_summary": results["final_context_summary"],
                "planning_history": [
                    {
                        "step_id": step["step_id"],
                        "agent_id": step["agent_id"],
                        "agent_role": step["agent_role"],
                        "context_length_before": step["context_length_before"],
                        "context_length_after": step["context_length_after"],
                        "compression_ratio": step["compression_ratio"],
                        "pruning_time": step["pruning_time"],
                        "inference_time": step["inference_time"],
                        "step_time": step["step_time"],
                        "structured_output": step["structured_output"],
                    }
                    for step in results["planning_history"]
                ],
            }
        else:
            # Multiple runs format
            total_time_all = sum(r["total_time"] for r in all_runs_results)
            total_pruning_time_all = sum(r["total_pruning_time"] for r in all_runs_results)
            total_inference_time_all = sum(r["total_inference_time"] for r in all_runs_results)
            
            json_results = {
                "mode": "baseline" if args.baseline else "erecap",
                "num_runs": args.num_runs,
                "task_description": all_runs_results[0]["task_description"],
                "num_agents": all_runs_results[0]["num_agents"],
                "total_wall_clock_time": total_time,
                "total_time": total_time_all,
                "average_time_per_run": total_time_all / args.num_runs,
                "total_pruning_time": total_pruning_time_all,
                "total_inference_time": total_inference_time_all,
                "final_context_summary": all_runs_results[-1]["final_context_summary"],
                "runs": [
                    {
                        "run_id": idx,
                        "total_time": r["total_time"],
                        "total_pruning_time": r["total_pruning_time"],
                        "total_inference_time": r["total_inference_time"],
                        "final_context_length": r["final_context_summary"]["total_context_length"],
                    }
                    for idx, r in enumerate(all_runs_results)
                ],
                "last_run_planning_history": [
                    {
                        "step_id": step["step_id"],
                        "agent_id": step["agent_id"],
                        "agent_role": step["agent_role"],
                        "context_length_before": step["context_length_before"],
                        "context_length_after": step["context_length_after"],
                        "compression_ratio": step["compression_ratio"],
                        "pruning_time": step["pruning_time"],
                        "inference_time": step["inference_time"],
                        "step_time": step["step_time"],
                    }
                    for step in all_runs_results[-1]["planning_history"]
                ],
            }
        
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

