import pickle, os
import numpy as np
import pdb
import zarr
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('--task_name', type=str, default='LiftBarrier',
                        help='The name of the task (e.g., LiftBarrier)')
    parser.add_argument('--agent_id', type=int, default=0,
                        help='The id of the agent (use -1 for global)')
    parser.add_argument('--load_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    agent_id = args.agent_id
    num = args.load_num
    
    # Handle global vs agent-specific data
    if agent_id == -1:
        load_dir = f'data/pkl_data/{task_name}_global'
        save_dir = f'data/zarr_data/{task_name}_global_{num}.zarr'
        is_global = True
    else:
        load_dir = f'data/pkl_data/{task_name}_Agent{agent_id}'
        save_dir = f'data/zarr_data/{task_name}_Agent{agent_id}_{num}.zarr'
        is_global = False
    
    total_count = 0

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays = []
    wrist_camera_arrays = []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    has_wrist_camera = False
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            # Head camera (always present)
            head_img = data['observation']['head_camera']['rgb']
            head_camera_arrays.append(head_img)
            
            # Wrist camera (only for agents, not global)
            if not is_global and 'wrist_camera' in data['observation']:
                wrist_img = data['observation']['wrist_camera']['rgb']
                wrist_camera_arrays.append(wrist_img)
                has_wrist_camera = True
            
            # Action data (only for agents, not global)
            if not is_global:
                action = data['endpose']
                joint_action = data['joint_action']
                action_arrays.append(action)
                state_arrays.append(joint_action)
                joint_action_arrays.append(joint_action)

            file_num += 1
            total_count += 1
            
        current_ep += 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    
    # Save wrist camera if present
    if has_wrist_camera and len(wrist_camera_arrays) > 0:
        wrist_camera_arrays = np.array(wrist_camera_arrays)
        wrist_camera_arrays = np.moveaxis(wrist_camera_arrays, -1, 1)  # NHWC -> NCHW
        wrist_camera_chunk_size = (100, *wrist_camera_arrays.shape[1:])
        zarr_data.create_dataset('wrist_camera', data=wrist_camera_arrays, chunks=wrist_camera_chunk_size, overwrite=True, compressor=compressor)
        print(f"Saved wrist_camera with shape: {wrist_camera_arrays.shape}")
    
    # Save action data only for agents (not global)
    if not is_global and len(action_arrays) > 0:
        action_arrays = np.array(action_arrays)
        state_arrays = np.array(state_arrays)
        joint_action_arrays = np.array(joint_action_arrays)
        
        action_chunk_size = (100, action_arrays.shape[1])
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        
        zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    print(f"Saved zarr data to: {save_dir}")
    print(f"  head_camera shape: {head_camera_arrays.shape}")
    if not is_global and len(action_arrays) > 0:
        print(f"  action shape: {action_arrays.shape}")

if __name__ == '__main__':
    main()