#!/usr/bin/env python3
"""
Unified data conversion CLI for RoboFactory datasets.

This script provides a single entry point for converting ZARR datasets
to various formats used by different VLA policies.

Usage:
    # Convert ZARR to LeRobot (for Pi0/Pi0.5)
    python -m robofactory.dataset.scripts.convert_data \\
        --input data/zarr_data/LiftBarrier-rf_Agent0_50.zarr \\
        --output data/lerobot_data \\
        --format lerobot \\
        --task LiftBarrier-rf \\
        --agent_id 0 \\
        --num_episodes 50
    
    # Convert ZARR to RLDS (for OpenVLA)
    python -m robofactory.dataset.scripts.convert_data \\
        --input data/zarr_data/LiftBarrier-rf_Agent0_50.zarr \\
        --output data/rlds_data \\
        --format rlds \\
        --task LiftBarrier-rf \\
        --agent_id 0
    
    # Batch convert all ZARR files in directory to RLDS
    python -m robofactory.dataset.scripts.convert_data \\
        --input data/zarr_data \\
        --output data/rlds_data \\
        --format rlds \\
        --batch
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_converter(format_name: str):
    """Get the appropriate converter class for the format."""
    from robofactory.dataset.converters import (
        ZarrToLeRobotConverter,
        ZarrToRLDSConverter,
    )
    
    converters = {
        'lerobot': ZarrToLeRobotConverter,
        'rlds': ZarrToRLDSConverter,
    }
    
    if format_name not in converters:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(converters.keys())}")
    
    return converters[format_name]()


def main():
    parser = argparse.ArgumentParser(
        description="Convert ZARR datasets to various VLA policy formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input ZARR file or directory (for batch mode)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=['lerobot', 'rlds'],
        required=True,
        help="Target format (lerobot for Pi0, rlds for OpenVLA)"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task name (e.g., 'LiftBarrier-rf')"
    )
    parser.add_argument(
        "--agent_id", "-a",
        type=int,
        help="Agent ID"
    )
    parser.add_argument(
        "--num_episodes", "-n",
        type=int,
        help="Number of episodes (for naming, auto-detected if not provided)"
    )
    parser.add_argument(
        "--global_zarr",
        type=str,
        default=None,
        help="Path to global camera ZARR (for LeRobot format)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom language instruction (auto-generated if not provided)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all ZARR files in input directory"
    )
    parser.add_argument(
        "--create_statistics",
        action="store_true",
        help="Also create statistics JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch:
        if not args.task:
            parser.error("--task is required when not in batch mode")
        if args.agent_id is None:
            parser.error("--agent_id is required when not in batch mode")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Handle batch mode
    if args.batch:
        if args.format == 'rlds':
            from robofactory.dataset.converters import batch_convert_zarr_to_rlds
            batch_convert_zarr_to_rlds(
                zarr_dir=args.input,
                output_dir=args.output,
            )
        else:
            print(f"Batch mode not yet supported for format: {args.format}")
            sys.exit(1)
        return
    
    # Single file conversion
    converter = get_converter(args.format)
    
    print(f"Converting {args.input} to {args.format} format...")
    print(f"  Task: {args.task}")
    print(f"  Agent ID: {args.agent_id}")
    
    try:
        if args.format == 'lerobot':
            output = converter.convert(
                input_path=args.input,
                output_path=args.output,
                task_name=args.task,
                agent_id=args.agent_id,
                num_episodes=args.num_episodes,
                language_instruction=args.instruction,
                global_zarr_path=args.global_zarr,
            )
        else:  # rlds
            output = converter.convert(
                input_path=args.input,
                output_path=args.output,
                task_name=args.task,
                agent_id=args.agent_id,
                language_instruction=args.instruction,
            )
        
        print(f"\n✅ Successfully converted to: {output}")
        
        # Create statistics if requested
        if args.create_statistics:
            from robofactory.dataset.converters.zarr_to_rlds import create_dataset_statistics
            import os
            
            stats_path = os.path.join(args.output, output if isinstance(output, str) else str(output), 'statistics.json')
            create_dataset_statistics(args.input, stats_path)
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

