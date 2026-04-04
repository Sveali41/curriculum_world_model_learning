import metaworld
import numpy as np
import gymnasium as gym

class CustomMetaWorldEnv(gym.Env):
    """
    Custom wrapper for Meta-World to support Curriculum / UED generation (Mini tasks),
    and zero-shot validation (Target tasks).
    """
    def __init__(self, env_name='sweep-into-v3', render_mode=None):
        super().__init__()
        self.env_name = env_name
        self.ml1 = metaworld.ML1(env_name)
        
        # Instantiate the unwrapped Meta-World environment
        if render_mode is not None:
            self.env = self.ml1.train_classes[env_name](render_mode=render_mode)
        else:
            self.env = self.ml1.train_classes[env_name]()
            
        # Set default task from ML1 (this is the Target task setting)
        self.default_task = self.ml1.train_tasks[0]
        self.env.set_task(self.default_task)
        
        # Generator customization parameters
        self.custom_obj_pos = None 
        self.custom_goal_pos = None
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def set_custom_layout(self, obj_pos=None, goal_pos=None):
        """
        供 UED Curriculum 生成器调用。根据输入连续坐标动态控制 Mini Task 难度。
        obj_pos: [x, y, z] 物体初始坐标
        goal_pos: [x, y, z] 目标终点坐标
        """
        if obj_pos is not None:
            self.custom_obj_pos = np.array(obj_pos, dtype=np.float32)
        else:
            self.custom_obj_pos = None
            
        if goal_pos is not None:
            self.custom_goal_pos = np.array(goal_pos, dtype=np.float32)
        else:
            self.custom_goal_pos = None

    def set_custom_layout_from_str(self, layout_str: str):
        """
        供 UED Curriculum 生成器调用（基于连续坐标字符串表征）。
        格式要求: "obj_x,obj_y,obj_z;goal_x,goal_y,goal_z"
        例如: "0.0,0.6,0.02;0.2,0.8,0.0"
        若传入 "" 或 "TARGET" 则恢复随机发牌。
        """
        if not layout_str or layout_str.upper() == "TARGET":
            self.set_custom_layout(None, None)
            return
            
        if ";" in layout_str:
            try:
                obj_str, goal_str = layout_str.split(";")
                obj_pos = [float(x.strip()) for x in obj_str.split(",")]
                goal_pos = [float(x.strip()) for x in goal_str.split(",")]
                self.set_custom_layout(obj_pos, goal_pos)
            except Exception as e:
                print(f"Warning: Failed to parse layout_str '{layout_str}'. Fallback to Target. Error: {e}")
                self.set_custom_layout(None, None)

    def reset(self, seed=None, options=None):
        """
        重置环境。如果设定了 custom_layout，则接管初始化并将环境强制设置在指定参数下。
        """
        # 1. 正常执行底层重置，使 MuJoCo 刷新并随机分配
        result = self.env.reset()
        
        # Old Gym API (MetaWorld v2 often returns just obs) vs Gymnasium API
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # 2. 如果存在外部干预参数，强制修改 Mujuco 底层对象位置和目标点
        if self.custom_obj_pos is not None or self.custom_goal_pos is not None:
            
            # (A) 修改目标点 (Target Goal)
            if self.custom_goal_pos is not None:
                self.env._target_pos = self.custom_goal_pos.copy()
                try:
                    # 将目标位置点同步到 MuJoCo 渲染 site (某些环境叫 'goal')
                    site_id = self.env.model.site_name2id('goal')
                    self.env.model.site_pos[site_id] = self.custom_goal_pos
                except Exception:
                    pass 
            
            # (B) 修改物体初始位置 (Object Position)
            if self.custom_obj_pos is not None:
                self.env.obj_init_pos = self.custom_obj_pos.copy()
                try:
                    self.env._set_obj_xyz(self.custom_obj_pos)
                except AttributeError:
                    pass
            
            # 3. 产生修改后的新观察值
            obs = self.env._get_obs()

        # 将解析出来的实体字典植入 info，方便 P2E 的 WM 使用
        entities = self._vector_to_entity_dict(obs)
        info['entities'] = entities
            
        return obs, info

    def _vector_to_entity_dict(self, obs_vector):
        """
        把连续的 39 维向量状态截断打包为 WM 需要的实体 Attention 格式。
        根据 SawyerXYZ 的一般定义：
        [0:4]   机器人末端执行器 xyz 和 夹爪状态开合度
        [4:18]  物体位置及四元数
        """
        target = self.env._target_pos if hasattr(self.env, '_target_pos') else np.zeros(3)
        return {
            "robot": obs_vector[0:4],
            "object": obs_vector[4:18], 
            "goal": target
        }
    
    def step(self, action):
        step_returns = self.env.step(action)
        
        # 兼容处理老 gym 的 4 个返回值，和最新 Gymnasium 的 5 个返回值
        if len(step_returns) == 4:
            obs, reward, done, info = step_returns
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_returns
            
        info['entities'] = self._vector_to_entity_dict(obs)
        
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(*args, **kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

if __name__ == "__main__":
    # 基础测试脚本
    print("Testing CustomMetaWorldEnv...")
    env = CustomMetaWorldEnv('sweep-into-v3', render_mode=None)
    
    print("\n--- [Target Task Mode] (Default Randomization) ---")
    obs, info = env.reset()
    print("Random Object Pos:", env.env.obj_init_pos)
    print("Random Goal Pos:", env.env._target_pos)
    
    print("\n--- [Mini Task Mode] (UED Curriculum Parameterized) ---")
    custom_obj = np.array([0.0, 0.6, 0.02])
    custom_goal = np.array([0.2, 0.8, 0.0])
    
    env.set_custom_layout(obj_pos=custom_obj, goal_pos=custom_goal)
    obs, info = env.reset()
    
    print("Customized Object Pos:", env.env.obj_init_pos)
    print("Customized Goal Pos:", env.env._target_pos)
    print("\n[Entity Extraction Format Example]")
    print(info['entities'])
    
    env.close()
