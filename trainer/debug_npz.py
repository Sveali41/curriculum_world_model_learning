import numpy as np
import os
import sys

def inspect_npz(filepath):
    print(f"\n--- Inspecting: {os.path.basename(filepath)} ---")
    data = np.load(filepath, allow_pickle=True)
    obs = data['a']
    print(f"Shape: {obs.shape}")
    
    # Check first frame
    frame = obs[0]
    # In MiniGrid, channel 0 is the object ID.
    # If HWC, it's [y, x, 0] or [x, y, 0].
    # If CHW, it's [0, y, x] or [0, x, y].
    
    if frame.ndim == 3:
        # Check where ID 10 (Agent) is
        if frame.shape[0] == 3: # CHW
            obj_layer = frame[0]
            print("Format: Likely CHW")
        else: # HWC
            obj_layer = frame[:, :, 0]
            print("Format: Likely HWC")
        
        agent_pos = np.argwhere(obj_layer == 10)
        if len(agent_pos) > 0:
            print(f"Agent found at index: {agent_pos[0]} (in spatial layer shape {obj_layer.shape})")
        else:
            print("Agent not found in first frame.")
            # Print unique values to see what's there
            print(f"Unique values in first layer: {np.unique(obj_layer)}")

if __name__ == "__main__":
    base_dir = "/home/siyao/phd_file/Research/rlPractice/Curriculum_world_model_learning/trainer/data/minigrid/target_tasks"
    
    target_file = os.path.join(base_dir, "target_task0_test_random.npz")
    p2e_file = os.path.join(base_dir, "p2e_target_task0.txt_c1.npz")
    
    if os.path.exists(target_file):
        inspect_npz(target_file)
    else:
        print(f"Target file not found: {target_file}")
        
    if os.path.exists(p2e_file):
        inspect_npz(p2e_file)
    else:
        print(f"P2E file not found: {p2e_file}")
