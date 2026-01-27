"""
ZARR to LeRobot format converter for Pi0/Pi0.5 training.

This module converts RoboFactory ZARR datasets to LeRobot format following
the openpi data pipeline conventions:
- Use LeRobot dataset format (not RLDS)
- Image keys following Pi0 requirements: "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
- State key: "state" (for proprio)
- Action key: "actions"
- Task key: "task" (for language instructions)

RoboFactory ZARR Data Structure:
- Agent ZARR: {TASK}_Agent{ID}_{NUM}.zarr
    - head_camera: side view (fixed position) → base_0_rgb
    - wrist_camera: gripper view (end-effector) → right_wrist_0_rgb
    
- Global ZARR: {TASK}_Agent{ID}_global_{NUM}.zarr
    - head_camera: overhead/global view (bird's eye) → left_wrist_0_rgb

Pi0 3-Camera Mapping:
- base_0_rgb      ← head_camera from Agent ZARR (side view)
- left_wrist_0_rgb ← head_camera from Global ZARR (overhead view)
- right_wrist_0_rgb ← wrist_camera from Agent ZARR (gripper view)
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import zarr
from tqdm import tqdm

try:
    # Try new lerobot structure (v0.4+)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        # Fallback to old structure (v0.2-0.3)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("Warning: LeRobot not installed. Please install with: pip install lerobot>=0.2.0")
        LeRobotDataset = None

from .base_converter import BaseDataConverter
from ..utils import process_image

# Import from shared module
from robofactory.policy.shared import get_task_instruction


class ZarrToLeRobotConverter(BaseDataConverter):
    """
    Converter for ZARR to LeRobot format (Pi0/Pi0.5).
    
    This converter produces datasets compatible with the openpi training
    pipeline, including proper 3-camera setup.
    
    Example:
        converter = ZarrToLeRobotConverter()
        output = converter.convert(
            input_path='data/zarr_data/LiftBarrier-rf_Agent0_50.zarr',
            output_path='data/lerobot_data',
            task_name='LiftBarrier-rf',
            agent_id=0,
            num_episodes=50,
        )
    """
    
    def get_output_format(self) -> str:
        return 'lerobot'
    
    def convert(
        self,
        input_path: str,
        output_path: str,
        task_name: str,
        agent_id: int,
        num_episodes: Optional[int] = None,
        language_instruction: Optional[str] = None,
        global_zarr_path: Optional[str] = None,
    ) -> str:
        """
        Convert ZARR dataset to LeRobot format.
        
        Args:
            input_path: Path to input ZARR dataset
            output_path: Output directory for LeRobot dataset
            task_name: Name of the task (e.g., 'LiftBarrier-rf')
            agent_id: Agent ID
            num_episodes: Number of episodes (for naming)
            language_instruction: Custom language instruction (auto-generated if None)
            global_zarr_path: Optional path to global camera ZARR
            
        Returns:
            Path to created LeRobot dataset
        """
        self.validate_input(input_path)
        
        return convert_zarr_to_lerobot(
            zarr_path=input_path,
            output_dir=output_path,
            task_name=task_name,
            agent_id=agent_id,
            num_episodes=num_episodes or self._count_episodes(input_path),
            language_instruction=language_instruction,
            global_zarr_path=global_zarr_path,
        )
    
    def _count_episodes(self, zarr_path: str) -> int:
        """Count number of episodes in ZARR dataset."""
        root = zarr.open(zarr_path, mode='r')
        if 'meta' in root and 'episode_ends' in root['meta']:
            return len(root['meta']['episode_ends'])
        return 1


def convert_zarr_to_lerobot(
    zarr_path: str,
    output_dir: str,
    task_name: str,
    agent_id: int,
    num_episodes: int,
    language_instruction: Optional[str] = None,
    global_zarr_path: Optional[str] = None,
) -> str:
    """
    Convert ZARR dataset to LeRobot format for Pi0/Pi0.5.
    
    Args:
        zarr_path: Path to input ZARR dataset
        output_dir: Output directory for LeRobot dataset
        task_name: Name of the task
        agent_id: Agent ID
        num_episodes: Number of episodes
        language_instruction: Custom language instruction
        global_zarr_path: Optional path to global camera ZARR
        
    Returns:
        Repository ID of created dataset
    """
    if LeRobotDataset is None:
        raise ImportError("LeRobot is required. Install with: pip install lerobot>=0.2.0")
    
    print(f"Loading ZARR data from {zarr_path}...")
    root = zarr.open(zarr_path, mode='r')
    
    data_group = root['data']
    meta_group = root['meta']
    
    # Load global ZARR if provided (contains overhead camera view)
    global_data_group = None
    if global_zarr_path and Path(global_zarr_path).exists():
        print(f"Loading global view ZARR from {global_zarr_path}...")
        global_root = zarr.open(global_zarr_path, mode='r')
        global_data_group = global_root['data']
        print(f"  ✓ Global ZARR loaded (overhead camera view)")
    elif global_zarr_path:
        print(f"  WARNING: Global ZARR not found at {global_zarr_path}, will duplicate head_camera")
    else:
        # Try to auto-detect global ZARR path
        global_data_group = _auto_detect_global_zarr(zarr_path)
    
    # Get episode boundaries
    episode_ends = np.array(meta_group['episode_ends'])
    
    # Check which camera format is used
    has_separate_cameras = 'head_camera' in data_group
    
    if has_separate_cameras:
        first_img = data_group['head_camera'][0]
        if first_img.shape[0] == 3:
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = True
        else:
            img_h, img_w = first_img.shape[0], first_img.shape[1]
            is_chw = False
        print(f"Found separate camera arrays with resolution {img_h}x{img_w}")
    else:
        first_img = data_group['img'][0]
        if first_img.shape[1] == 3:
            img_h, img_w = first_img.shape[2], first_img.shape[3]
            is_chw = True
        else:
            img_h, img_w = first_img.shape[1], first_img.shape[2]
            is_chw = False
        print(f"Found img array with resolution {img_h}x{img_w}")
    
    # Get action and state dimensions
    action_dim = data_group['action'].shape[-1]
    state_key = 'state' if 'state' in data_group else 'agent_pos'
    state_dim = data_group[state_key].shape[-1]
    
    # Prepare output
    repo_id = f"{task_name}_Agent{agent_id}_{num_episodes}"
    output_path = Path(output_dir) / repo_id
    
    # Remove existing output if it exists
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Clean HuggingFace cache for this repo to avoid conflicts
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    if cache_path.exists():
        print(f"Cleaning cache at {cache_path}")
        shutil.rmtree(cache_path)
    
    print(f"Creating LeRobot dataset: {repo_id}")
    print(f"  Action dim: {action_dim}, State dim: {state_dim}")
    
    # Create LeRobot dataset with Pi0's 3-camera format
    # Use image format (not video) - images stored in parquet files
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "base_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "left_wrist_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_0_rgb": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=0,  # Use 0 to write synchronously (avoids corruption)
        image_writer_processes=0,
    )
    
    # Set default language instruction
    if language_instruction is None:
        language_instruction = get_task_instruction(task_name, policy_type='detailed')
    
    print(f"Converting {len(episode_ends)} episodes")
    print(f"Language instruction: '{language_instruction}'")
    
    # Convert episodes
    start_idx = 0
    for ep_idx, end_idx in enumerate(tqdm(episode_ends, desc="Converting episodes")):
        for t in range(start_idx, end_idx):
            head_img, global_img, wrist_img = _extract_images(
                data_group, global_data_group, t,
                has_separate_cameras, is_chw
            )
            
            # Ensure uint8 format
            head_img = process_image(head_img)
            global_img = process_image(global_img)
            wrist_img = process_image(wrist_img)
            
            # Get state/action
            state_data = data_group[state_key][t]
            action_data = data_group['action'][t]
            
            frame_data = {
                "base_0_rgb": head_img,
                "left_wrist_0_rgb": global_img,
                "right_wrist_0_rgb": wrist_img,
                "state": state_data.astype(np.float32),
                "actions": action_data.astype(np.float32),
                "task": language_instruction,
            }
            
            dataset.add_frame(frame_data)
        
        dataset.save_episode()
        start_idx = end_idx
    
    # CRITICAL: Properly finalize the dataset to avoid corrupted parquet files
    # The parquet footer with "PAR1" magic bytes won't be written until files are closed
    
    # 1. Wait for any async writers
    if hasattr(dataset, 'image_writer') and dataset.image_writer is not None:
        try:
            print("  Waiting for image writer to finish...")
            dataset.image_writer.wait_until_done()
        except Exception as e:
            print(f"  Note: image_writer.wait_until_done() raised: {e}")
    
    if hasattr(dataset, 'video_writer') and dataset.video_writer is not None:
        try:
            print("  Waiting for video writer to finish...")
            dataset.video_writer.wait_until_done()
        except Exception as e:
            print(f"  Note: video_writer.wait_until_done() raised: {e}")
    
    # 2. Call consolidate to finalize the dataset
    if hasattr(dataset, 'consolidate'):
        try:
            print("  Consolidating dataset...")
            dataset.consolidate()
        except Exception as e:
            print(f"  Note: consolidate() raised: {e}")
    
    # 3. IMPORTANT: Delete the dataset object to force file handles to close
    # This ensures parquet footers are written before we move files
    import gc
    import time
    
    print("  Finalizing files...")
    del dataset
    gc.collect()
    
    # 4. Wait a moment for filesystem to sync
    time.sleep(2)
    
    print(f"✅ Successfully converted {len(episode_ends)} episodes to LeRobot format")
    
    # Move dataset from cache to output directory
    _move_from_cache(repo_id, output_dir, output_path)
    
    # Debug: List what files were created
    if output_path.exists():
        data_dir = output_path / "data"
        if data_dir.exists():
            print(f"   Dataset structure in {data_dir}:")
            for item in data_dir.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"     {item.relative_to(output_path)}: {size} bytes")
    
    # Consolidate and fix parquet files (optional step, won't fail conversion)
    actual_total_frames = _consolidate_parquet_files(output_path)
    
    # Update info.json if needed
    _update_info_json(output_path, actual_total_frames)
    
    # Create episodes metadata
    _create_episodes_metadata(output_path, episode_ends, language_instruction, actual_total_frames)
    
    print(f"   Output: {output_path}")
    return repo_id


def _auto_detect_global_zarr(zarr_path: str):
    """Auto-detect global ZARR path based on agent ZARR path."""
    zarr_stem = Path(zarr_path).stem
    
    # Pattern: {task}_Agent{id}_{num} -> {task}_global_{num}
    match = re.match(r'(.+)_Agent\d+_(\d+)$', zarr_stem)
    if match:
        task_name_part = match.group(1)
        num_part = match.group(2)
        global_zarr_name = f"{task_name_part}_global_{num_part}.zarr"
        auto_global_path = Path(zarr_path).parent / global_zarr_name
        if auto_global_path.exists():
            print(f"Auto-detected global ZARR: {auto_global_path}")
            global_root = zarr.open(str(auto_global_path), mode='r')
            return global_root['data']
        else:
            print(f"  INFO: No global ZARR found at {auto_global_path}, will duplicate head_camera")
    else:
        print(f"  INFO: Could not parse ZARR name pattern, will duplicate head_camera")
    return None


def _extract_images(data_group, global_data_group, t, has_separate_cameras, is_chw):
    """Extract and format images from ZARR data."""
    if has_separate_cameras:
        head_img = data_group['head_camera'][t]
        wrist_img = data_group['wrist_camera'][t]
        
        if global_data_group is not None and 'head_camera' in global_data_group:
            global_img = global_data_group['head_camera'][t]
        else:
            global_img = head_img.copy()
        
        if is_chw:
            head_img = np.transpose(head_img, (1, 2, 0))
            global_img = np.transpose(global_img, (1, 2, 0))
            wrist_img = np.transpose(wrist_img, (1, 2, 0))
    else:
        img_t = data_group['img'][t]
        
        if is_chw:
            head_img = np.transpose(img_t[0], (1, 2, 0))
            wrist_img = np.transpose(img_t[2] if img_t.shape[0] > 2 else img_t[1], (1, 2, 0))
        else:
            head_img = img_t[0]
            wrist_img = img_t[2] if img_t.shape[0] > 2 else img_t[1]
        
        if global_data_group is not None:
            if 'head_camera' in global_data_group:
                global_img = global_data_group['head_camera'][t]
                if is_chw:
                    global_img = np.transpose(global_img, (1, 2, 0))
            elif 'img' in global_data_group:
                global_img_t = global_data_group['img'][t]
                if is_chw:
                    global_img = np.transpose(global_img_t[0], (1, 2, 0))
                else:
                    global_img = global_img_t[0]
            else:
                global_img = head_img.copy()
        else:
            if is_chw:
                global_img = np.transpose(img_t[1] if img_t.shape[0] > 2 else img_t[0], (1, 2, 0))
            else:
                global_img = img_t[1] if img_t.shape[0] > 2 else img_t[0]
    
    return head_img, global_img, wrist_img


def _move_from_cache(repo_id: str, output_dir: str, output_path: Path):
    """Move dataset from HuggingFace cache to output directory."""
    import time
    import os
    
    cache_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    
    if cache_path.exists():
        # First, verify the parquet files are valid (have proper footer)
        parquet_files = list(cache_path.rglob("*.parquet"))
        for pf in parquet_files:
            if not _verify_parquet_file(pf):
                print(f"   WARNING: Parquet file may be incomplete: {pf}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"   Moving dataset from {cache_path} to {output_path}")
        if output_path.exists():
            shutil.rmtree(output_path)
        
        # Wait briefly to ensure all file handles are released
        time.sleep(1.0)
        
        # Use copy instead of move to be safer, then delete source
        try:
            shutil.copytree(str(cache_path), str(output_path))
            # Sync filesystem
            if hasattr(os, 'sync'):
                os.sync()
            # Remove source after successful copy
            shutil.rmtree(str(cache_path))
        except Exception as e:
            print(f"   Warning: Copy failed ({e}), trying move...")
            if output_path.exists():
                shutil.rmtree(output_path)
            shutil.move(str(cache_path), str(output_path))


def _verify_parquet_file(parquet_path: Path) -> bool:
    """Verify parquet file has proper magic bytes in header and footer."""
    try:
        with open(parquet_path, 'rb') as f:
            # Check header (first 4 bytes should be 'PAR1')
            header = f.read(4)
            if header != b'PAR1':
                print(f"   Invalid parquet header in {parquet_path.name}: {header}")
                return False
            
            # Check footer (last 4 bytes should be 'PAR1')
            f.seek(-4, 2)  # Seek to 4 bytes from end
            footer = f.read(4)
            if footer != b'PAR1':
                print(f"   Invalid parquet footer in {parquet_path.name}: {footer}")
                return False
            
        return True
    except Exception as e:
        print(f"   Error verifying {parquet_path.name}: {e}")
        return False


def _consolidate_parquet_files(output_path: Path):
    """Count total frames from all parquet files WITHOUT merging.
    
    LeRobot v3.0+ natively supports reading multiple parquet files via
    the data_path template in info.json. Merging is not only unnecessary
    but causes memory issues for large datasets during training.
    
    The HuggingFace Datasets library underneath LeRobot automatically
    discovers and reads all parquet files matching the data_path pattern.
    """
    import pyarrow.parquet as pq
    
    data_chunk_dir = output_path / "data" / "chunk-000"
    actual_total_frames = 0
    
    if not data_chunk_dir.exists():
        print(f"   Note: No data/chunk-000 directory found, skipping frame count")
        return None
    
    parquet_files = sorted(data_chunk_dir.glob("*.parquet"))
    if len(parquet_files) == 0:
        print(f"   Note: No parquet files found in {data_chunk_dir}, skipping frame count")
        return None
    
    # Count frames from all parquet files WITHOUT merging
    print(f"   Counting frames from {len(parquet_files)} parquet file(s)...")
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            actual_total_frames += len(table)
        except Exception as e:
            print(f"   Warning: Could not read {pf.name}: {e}, skipping")
    
    print(f"   ✓ Total frames: {actual_total_frames} across {len(parquet_files)} file(s)")
    return actual_total_frames


def _update_info_json(output_path: Path, actual_total_frames):
    """Update info.json with actual total frames."""
    if actual_total_frames is not None:
        info_path = output_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            if info.get('total_frames') != actual_total_frames:
                print(f"   Updating info.json: total_frames {info.get('total_frames')} -> {actual_total_frames}")
                info['total_frames'] = actual_total_frames
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=4)


def _create_episodes_metadata(output_path: Path, episode_ends, language_instruction, actual_total_frames):
    """Create meta/episodes directory with parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    episodes_chunk_dir = output_path / "meta" / "episodes" / "chunk-000"
    episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
    
    episode_data = []
    start_idx = 0
    
    # Use episode_ends directly if we couldn't determine actual_total_frames
    # or if actual_total_frames matches expected
    if actual_total_frames is None or actual_total_frames >= episode_ends[-1]:
        for ep_idx, end_idx in enumerate(episode_ends):
            episode_length = end_idx - start_idx
            episode_data.append({
                "episode_index": ep_idx,
                "tasks": [language_instruction],
                "length": episode_length,
                "dataset_from_index": start_idx,
                "dataset_to_index": end_idx,
            })
            start_idx = end_idx
    else:
        # Data was lost - recalculate episode boundaries based on available data
        print(f"   Recalculating episode boundaries (actual frames: {actual_total_frames}, expected: {episode_ends[-1]})")
        remaining_frames = actual_total_frames
        for ep_idx, end_idx in enumerate(episode_ends):
            original_length = end_idx - start_idx
            if remaining_frames <= 0:
                break
            actual_length = min(original_length, remaining_frames)
            actual_end = start_idx + actual_length
            episode_data.append({
                "episode_index": ep_idx,
                "tasks": [language_instruction],
                "length": actual_length,
                "dataset_from_index": start_idx,
                "dataset_to_index": actual_end,
            })
            remaining_frames -= actual_length
            start_idx = actual_end
    
    table = pa.Table.from_pylist(episode_data)
    parquet_path = episodes_chunk_dir / "file-000.parquet"
    pq.write_table(table, parquet_path)
    print(f"   ✓ Created meta/episodes/chunk-000/file-000.parquet with {len(episode_data)} episodes")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ZARR to LeRobot format for Pi0")
    parser.add_argument("--zarr_path", type=str, required=True)
    parser.add_argument("--global_zarr_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/lerobot_data")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--agent_id", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    
    args = parser.parse_args()
    
    convert_zarr_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        task_name=args.task_name,
        agent_id=args.agent_id,
        num_episodes=args.num_episodes,
        global_zarr_path=args.global_zarr_path,
    )

