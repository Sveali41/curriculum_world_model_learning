import gymnasium as gym
import numpy as np
import re
import sys
import os

# Ensure the directory is in sys.path to allow relative-like imports when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bipedal_walker_custom import BipedalWalkerCustom

class CustomBipedalEnv(gym.Wrapper):
    """
    Advanced BipedalWalker Wrapper for UED.
    Supports structured layout strings with physical parameters.
    Format example: "R1.3 G30 S1.5 G10 P4 G10 T2 G30"
    Types: R (Roughness), G (Grass), S (Stump), P (Pit), T (Stairs)
    """
    def __init__(self, env_name='BipedalWalker-v3', render_mode=None):
        # We instantiate our custom engine instead of the gym default
        env = BipedalWalkerCustom(render_mode=render_mode, hardcore=True)
        super().__init__(env)
        
        self.layout_list = None
        self.layout_str = None

    def set_custom_layout_from_str(self, layout_str: str):
        """
        Parses a layout string into a list of (type, param) tuples.
        Example: "R1.5 G30 S1.5 P4 T2"
        -> [('R', 1.5), ('G', 30.0), ('S', 1.5), ('P', 4.0), ('T', 2.0)]
        """
        if not layout_str or layout_str.upper() == "TARGET":
            self.layout_list = None
            self.layout_str = None
            self.env.unwrapped.custom_layout = None
            return

        # Regular expression to extract type and numeric parameter
        # Matches: R1.2, G20, S1.5, T-3, P4
        tokens = re.findall(r'([RGSPT])(-?\d+\.?\d*)', layout_str.replace(" ", ""))
        self.layout_list = [(t, float(v)) for t, v in tokens]
        self.layout_str = layout_str
        
        # Inject into the unwrapped engine
        self.env.unwrapped.custom_layout = self.layout_list

    def reset(self, seed=None, options=None):
        # The custom _generate_terrain in BipedalWalkerCustom will use self.custom_layout
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Observation splitting for Attention WM
        info['entities'] = self._vector_to_entity_dict(obs)
        return obs, info

    def _vector_to_entity_dict(self, obs_vector):
        """
        Splits the 24-dim state for Attention-based World Models.
        Hull/Joints (14) + Lidar (10)
        """
        return {
            "hull_and_joints": obs_vector[0:14],
            "lidar_rays": obs_vector[14:24]     
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['entities'] = self._vector_to_entity_dict(obs)
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    import time
    print("Testing Advanced CustomBipedalEnv with Physical Parameters...")
    try:
        # 开启 "human" 渲染模式
        env = CustomBipedalEnv(render_mode="human")
        
        def run_visual_episode(env, layout_str, frames=200):
            print(f"\n--- [Task] {layout_str} ---")
            env.set_custom_layout_from_str(layout_str)
            obs, info = env.reset()
            print("Layout Applied: ", env.layout_list)
            
            for _ in range(frames):
                # 在没有训练好的策略时，我们随便给点向前的力矩（或者随机动作）来看看地形长什么样
                action = env.action_space.sample() 
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 减慢一点帧率，方便肉眼看清地形积木
                time.sleep(0.02)
                
                if terminated or truncated:
                    break
        
        # # 演示 1: stump mini env
        # run_visual_episode(env, "G20 S3.0 ")
        
        # # 演示 2: pit mini env
        # run_visual_episode(env, "G20 P4.0")
        
        # 演示 3: stairs mini env
        run_visual_episode(env, "G20 T4 G40")

        # # 演示 4: target env
        # run_visual_episode(env, "R5")
        
        env.close()
    except Exception as e:
        print(f"\n>>> Error: {e}")
        print("Please ensure Box2D is installed: pip install swig && pip install gymnasium[box2d]")
