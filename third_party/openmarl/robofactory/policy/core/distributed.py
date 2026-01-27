"""
Distributed training utilities for VLA policies.

This module provides common utilities for distributed data parallel (DDP) training
that all policy implementations can use, ensuring consistent distributed training
behavior across the codebase.

Patterns extracted from:
- Pi0Workspace._setup_ddp: NCCL backend, environment-based initialization
- OpenVLAWorkspace.__init__: Rank handling, device selection
- robotworkspace.py distributed utilities: Barrier, tensor reduction
"""

import os
from typing import Tuple
import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[bool, int, torch.device]:
    """
    Setup distributed training if launched with torchrun.
    
    Initializes the process group and sets up CUDA devices based on
    environment variables set by torchrun/torchrun.
    
    Environment variables used:
        WORLD_SIZE: Total number of processes
        RANK: Global rank of this process
        LOCAL_RANK: Local rank on this node
    
    Returns:
        Tuple of (is_distributed, local_rank, device)
        - is_distributed: True if running in distributed mode
        - local_rank: Local rank on this node (0 if not distributed)
        - device: torch.device for this process
    
    Example:
        distributed, local_rank, device = setup_distributed()
        model = model.to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank])
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    
    if is_distributed and not dist.is_initialized():
        # Choose backend based on CUDA availability
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        # Initialize process group using environment variables
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )
        
        # Set debug environment for better error messages
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    # Get local rank (for multi-node, this is rank within the node)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    if is_distributed and is_main_process():
        print(f"Initialized distributed training:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {backend if is_distributed else 'N/A'}")
        print(f"  Device: {device}")
    
    return is_distributed, local_rank, device


def cleanup_distributed():
    """
    Clean up distributed training.
    
    Should be called at the end of training to properly shut down
    the distributed process group.
    """
    if dist.is_initialized():
        # Synchronize all processes before cleanup
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).
    
    Use this to guard logging, checkpointing, and other operations
    that should only happen once.
    
    Returns:
        True if this is the main process (rank 0) or not in distributed mode
    
    Example:
        if is_main_process():
            print("Training started!")
            wandb.init(...)
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """
    Get the global rank of this process.
    
    Returns:
        Global rank (0 if not in distributed mode)
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Get the local rank of this process on its node.
    
    Returns:
        Local rank from LOCAL_RANK environment variable
    """
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def get_world_size() -> int:
    """
    Get the total number of processes.
    
    Returns:
        World size (1 if not in distributed mode)
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation - "mean", "sum", "min", "max"
        
    Returns:
        Reduced tensor
        
    Example:
        loss = reduce_tensor(loss, op="mean")
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone to avoid modifying original
    rt = tensor.clone()
    
    # Select reduction operation
    if op == "mean":
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / get_world_size()
    elif op == "sum":
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    elif op == "min":
        dist.all_reduce(rt, op=dist.ReduceOp.MIN)
    elif op == "max":
        dist.all_reduce(rt, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")
    
    return rt


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def barrier():
    """
    Synchronize all processes.
    
    Blocks until all processes reach this point. Useful for ensuring
    all processes have completed a step before proceeding.
    """
    if dist.is_initialized():
        dist.barrier()


def gather_objects(obj, dst: int = 0):
    """
    Gather arbitrary Python objects from all ranks to dst rank.
    
    Args:
        obj: Python object to gather
        dst: Destination rank
        
    Returns:
        List of objects from all ranks (only on dst rank), None on other ranks
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    rank = get_rank()
    
    if rank == dst:
        gathered = [None] * world_size
    else:
        gathered = None
    
    dist.gather_object(obj, gathered, dst=dst)
    
    return gathered


def all_gather_objects(obj):
    """
    All-gather arbitrary Python objects from all ranks.
    
    Args:
        obj: Python object to gather
        
    Returns:
        List of objects from all ranks
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    
    return gathered

