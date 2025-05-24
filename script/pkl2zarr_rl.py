import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2
import zarr.meta

import zarr
import numpy as np
from pathlib import Path

def process_done(data):
    # 获取episode结束索引
    episode_ends = data['meta']['episode_ends'][:]
    
    # 处理done字段
    if 'done' not in data['data']:
        # 创建初始化为0的done数组
        total_steps = data['data']['action'].shape[0]
        done = np.zeros(total_steps, dtype=np.int32)
        
        # 标记episode终止位置
        for end_idx in episode_ends:
            if end_idx <= total_steps:
                done[end_idx-1] = 1
            else:
                print(f"Warning: Episode end index {end_idx} exceeds dataset size {total_steps}")
        
        # 将done数组写入数据集（需要写入权限时使用）
        # 注意：如果数据集是只读模式，这里应该在内存中创建副本
        data['data']['done'] = done
        
    return data

def process_next_obs(data):
    episode_ends = data['meta']['episode_ends'][:]  # 获取所有episode结束位置
    head_cam = data['data']['head_camera']  # 原始头摄像头数据
    right_cam = data['data']['right_camera']
    next_head_cam = np.zeros_like(head_cam)  # 初始化next_head_cam
    next_right_cam = np.zeros_like(right_cam)  # 初始化next_head_cam
    state = data['data']['state']
    next_state = np.zeros_like(state)
    
    start_idx = 0
    for end_idx in episode_ends:
        # 处理当前episode内的数据
        episode_slice = slice(start_idx, end_idx)
        
        # 常规平移：当前episode内，next_head_cam[t] = head_cam[t+1]
        if end_idx - start_idx > 1:  # 确保episode长度>1
            next_head_cam[episode_slice][:-1] = head_cam[episode_slice][1:]
            next_right_cam[episode_slice][:-1] = right_cam[episode_slice][1:]
            next_state[episode_slice][:-1] = state[episode_slice][1:]
        
        # episode最后一个时间步的next_head_cam设为0（终止状态）
        next_head_cam[end_idx - 1] = 0  # -1因为end_idx是exclusive的
        next_right_cam[end_idx - 1] = 0  # -1因为end_idx是exclusive的
        next_state[end_idx - 1] = 0
        
        start_idx = end_idx  # 移动到下一个episode
    
    # 处理最后一个episode之后的数据（如果有）
    if start_idx < len(head_cam):
        next_head_cam[start_idx:-1] = head_cam[start_idx+1:]
        next_head_cam[-1] = 0  # 整个数据集的最后一个时间步
        next_right_cam[start_idx:-1] = right_cam[start_idx+1:]
        next_right_cam[-1] = 0  # 整个数据集的最后一个时间步
        next_state[start_idx:-1] = state[start_idx+1:]
        next_state[-1] = 0  # 整个数据集的最后一个时间步
    
    data['data']['next_head_camera'] = next_head_cam
    data['data']['next_right_camera'] = next_right_cam
    data['data']['next_state'] = next_state
    return data

def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='block_hammer_beat',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    head_camera_type = args.head_camera_type
    load_dir = f'data/{task_name}_{head_camera_type}_{num}_pkl'
    
    total_count = 0

    save_dir = f'./policy/Diffusion-Policy/data/{task_name}_{head_camera_type}_{num}.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = [], [], [], []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    reward_arrays = []
    apple_pose_arrays = []
    cabinet_pose_arrays = []

    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            head_img = data['observation']['head_camera']['rgb']
            right_img = data['observation']['right_camera']['rgb']
            action = data['endpose']
            joint_action = data['joint_action']
            reward = data['reward']  # add reward
            # apple_pose = data['apple_pose'] # add pose
            # apple_pose = np.concatenate([apple_pose.p, apple_pose.q])   # Flatten to [x, y, z, qx, qy, qz, qw]
            # cabinet_pose = data['cabinet_pose'] # add pose
            # cabinet_pose = np.concatenate([cabinet_pose.p, cabinet_pose.q]) # Flatten to [x, y, z, qx, qy, qz, qw]

            head_camera_arrays.append(head_img)
            right_camera_arrays.append(right_img)
            action_arrays.append(action)
            state_arrays.append(joint_action)
            joint_action_arrays.append(joint_action)
            reward_arrays.append(reward)     # add reward
            # apple_pose_arrays.append(apple_pose)  # 
            # cabinet_pose_arrays.append(cabinet_pose)  #

            file_num += 1
            total_count += 1
            
        current_ep += 1

        episode_ends_arrays.append(total_count)
    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    right_camera_arrays = np.array(right_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    reward_arrays = np.array(reward_arrays) # add reward
    apple_pose_arrays = np.array(apple_pose_arrays)  # 转换为numpy数组
    cabinet_pose_arrays = np.array(cabinet_pose_arrays)  # 转换为numpy数组

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    right_camera_arrays = np.moveaxis(right_camera_arrays, -1, 1)  # NHWC -> NCHW
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    right_camera_chunk_size = (100, *right_camera_arrays.shape[1:])
    reward_chunk_size = (100,)  # add reward
    # apple_pose_chunk_size = (100, 7)  # apple_pose
    # cabinet_pose_chunk_size = (100, 7)  # cabinet_pose
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('right_camera', data=right_camera_arrays, chunks=right_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('reward', data=reward_arrays, chunks=reward_chunk_size, dtype='float32', overwrite=True, compressor=compressor)    # add reward
    # zarr_data.create_dataset('apple_pose', data=apple_pose_arrays, chunks=apple_pose_chunk_size, dtype='float32', overwrite=True, compressor=compressor)  # 存储apple_pose
    # zarr_data.create_dataset('cabinet_pose', data=cabinet_pose_arrays, chunks=cabinet_pose_chunk_size, dtype='float32', overwrite=True, compressor=compressor)  # 存储cabinet_pose
    
    # process zarr and add some keys
    print("processing zarr and add some keys")
    data = zarr.open(save_dir, mode='a')  # 'a'模式允许修改
    data = process_done(data)
    data = process_next_obs(data)
if __name__ == '__main__':
    main()

