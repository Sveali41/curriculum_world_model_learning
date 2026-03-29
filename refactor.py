import re

with open('generator/generator_interface.py', 'r') as f:
    code = f.read()

# 1. Imports
code = code.replace(
    'from generator.env_designer import PCGSeeder, task_placer',
    'from generator.minigrid_env_designer import PCGSeeder as MinigridPCGSeeder, task_placer as minigrid_task_placer\nfrom generator.crafter_env_designer import CrafterPCGSeeder, CRAFTER_ACTION_MAP, CRAFTER_OBJ_MAP'
)

# 2. ACTION_TABLE to ACTION_TABLE_MINIGRID
code = code.replace('ACTION_TABLE = {', 'ACTION_TABLE_MINIGRID = {')

# 3. __init__ modifications
old_init = """        if agent_type == 'random':
            print("[Generator] Using Random Agent")
            self.ppo = RandomGeneratorAgent(num_actions=len(ACTION_TABLE_MINIGRID), device=device)
        else:
            self.ppo = GeneratorPPO(
                context_dim=hparams.context_dim,
                num_actions=len(ACTION_TABLE_MINIGRID),
                his_emb_dim=hparams.his_emb_dim,
                top_k_features=hparams.ctx_top_k_features,
                ablation_type=self.cfg.ablation.type 
            )
        
        self.map_height = hparams.map_height
        self.map_width = hparams.map_width
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(self.map_height, self.map_width, self.div_k)
        self.max_edits = hparams.max_edits
        self.seeder = PCGSeeder(height=self.map_height, width=self.map_width)
        
        self.OBJ_START = OBJECT_TO_IDX["agent"]
        self.OBJ_GOAL = OBJECT_TO_IDX["goal"]
        self.OBJ_EMPTY = OBJECT_TO_IDX["empty"]"""

new_init = """        self.is_crafter = getattr(cfg.attention_model, "env_type", "") == "crafter"
        self.ACTION_TABLE = CRAFTER_ACTION_MAP if self.is_crafter else ACTION_TABLE_MINIGRID
        
        if agent_type == 'random':
            print("[Generator] Using Random Agent")
            self.ppo = RandomGeneratorAgent(num_actions=len(self.ACTION_TABLE), device=device)
        else:
            self.ppo = GeneratorPPO(
                context_dim=hparams.context_dim,
                num_actions=len(self.ACTION_TABLE),
                his_emb_dim=hparams.his_emb_dim,
                top_k_features=hparams.ctx_top_k_features,
                ablation_type=self.cfg.ablation.type 
            )
        
        # 对于 Crafter，画布向下扩展1行作为背包控制板
        self.map_height = hparams.map_height + 1 if self.is_crafter else hparams.map_height
        self.map_width = hparams.map_width
        self.div_k = hparams.div_k
        self.diversity = DiversityModule(self.map_height, self.map_width, self.div_k)
        self.max_edits = hparams.max_edits
        
        if self.is_crafter:
            self.seeder = CrafterPCGSeeder(height=self.map_height, width=self.map_width)
            self.OBJ_START = CRAFTER_OBJ_MAP["agent"]
            self.OBJ_GOAL = CRAFTER_OBJ_MAP["water"]  # 用于掩码保护边界
            self.OBJ_EMPTY = CRAFTER_OBJ_MAP["grass"]
        else:
            self.seeder = MinigridPCGSeeder(height=self.map_height, width=self.map_width)
            self.OBJ_START = OBJECT_TO_IDX["agent"]
            self.OBJ_GOAL = OBJECT_TO_IDX["goal"]
            self.OBJ_EMPTY = OBJECT_TO_IDX["empty"]"""

code = code.replace(old_init, new_init)

# 4. step() seeder modifications
old_seeder_call = """            grid = self.seeder.generate(z=z)
            min_dim = min(self.map_height, self.map_width)
            adaptive_ratio = 0.85 if min_dim <= 10 else 0.4
            grid, _ = task_placer(grid, min_dist_ratio=adaptive_ratio)"""

new_seeder_call = """            grid = self.seeder.generate(z=z)
            if not self.is_crafter:
                min_dim = min(self.map_height, self.map_width)
                adaptive_ratio = 0.85 if min_dim <= 10 else 0.4
                grid, _ = minigrid_task_placer(grid, min_dist_ratio=adaptive_ratio)"""

code = code.replace(old_seeder_call, new_seeder_call)

# 5. _immutable_mask wrapper
old_mask = """        mask[:, 0, :] = 1.0
        mask[:, -1, :] = 1.0
        mask[:, :, 0] = 1.0
        mask[:, :, -1] = 1.0"""
new_mask = """        mask[:, 0, :] = 1.0
        if self.is_crafter:
            mask[:, -2, :] = 1.0 # Protect physical map bottom border (H-2)
            # The row -1 is the inventory. We do NOT mask it because we WANT generator to edit it!
            # But we mask the corners of inventory to save computing
            mask[:, -1, 0] = 1.0
            mask[:, -1, -1] = 1.0
        else:
            mask[:, -1, :] = 1.0
        mask[:, :, 0] = 1.0
        mask[:, :, -1] = 1.0"""
code = code.replace(old_mask, new_mask)


# 5. _apply_action logic override
import re
apply_action_pattern = re.compile(r"def _apply_action\(self, base_obj_map, act\):.*?return obj, color", re.DOTALL)

new_apply_action = """def _apply_action(self, base_obj_map, act):
        if self.is_crafter:
            H, W = base_obj_map.shape
            obj = base_obj_map.copy()
            # Crafter doesn't use color map from generator, just return zeros to keep API compatible
            color = np.zeros_like(obj)
            immutable = (obj == self.OBJ_START) | (obj == self.OBJ_GOAL)
            
            for i in range(H):
                for j in range(W):
                    if immutable[i, j]: continue
                    a = act[i, j]
                    if a == 0: continue
                    
                    if a in self.ACTION_TABLE:
                        act_val = self.ACTION_TABLE[a]
                        # 拦截针对最后一行(Inventory Row)的操作和普通操作
                        if isinstance(act_val, str) and i == H - 1:
                            if act_val == "inventory_add_1":
                                obj[i, j] += 1
                            elif act_val == "inventory_add_2":
                                obj[i, j] += 5
                        elif not isinstance(act_val, str) and i < H - 1:
                            # 正常的地形修改
                            obj[i, j] = act_val
                            
            return obj, color
            
        else:
            H, W = base_obj_map.shape
            obj = base_obj_map.copy()
            
            MAX_OBJ_ID = max(OBJECT_TO_IDX.values())
            default_color_map = np.zeros(MAX_OBJ_ID + 1, dtype=np.int64)
            default_color_map[OBJECT_TO_IDX["wall"]] = COLOR_TO_IDX["grey"]
            default_color_map[OBJECT_TO_IDX["door"]] = COLOR_TO_IDX["yellow"]
            default_color_map[OBJECT_TO_IDX["key"]] = COLOR_TO_IDX["yellow"]
            default_color_map[OBJECT_TO_IDX["ball"]] = COLOR_TO_IDX["red"]
            default_color_map[OBJECT_TO_IDX["box"]] = COLOR_TO_IDX["yellow"]
            default_color_map[OBJECT_TO_IDX["goal"]] = COLOR_TO_IDX["green"]
            default_color_map[OBJECT_TO_IDX["lava"]] = COLOR_TO_IDX["red"]
            
            color = default_color_map[np.clip(obj, 0, MAX_OBJ_ID)]
            immutable = (obj == self.OBJ_START) | (obj == self.OBJ_GOAL)

            for i in range(H):
                for j in range(W):
                    if immutable[i, j]: continue
                    a = act[i, j]
                    if a == 0: continue
                    obj_type, color_name = self.ACTION_TABLE[a]
                    if obj_type == "key":
                        obj[i, j] = OBJECT_TO_IDX["key"]; color[i, j] = COLOR_TO_IDX[color_name]
                    elif obj_type == "door":
                        obj[i, j] = OBJECT_TO_IDX["door"]; color[i, j] = COLOR_TO_IDX[color_name]
                    elif obj_type == "lava":
                        obj[i, j] = OBJECT_TO_IDX["lava"]
                    elif obj_type == "empty":
                        obj[i, j] = OBJECT_TO_IDX["empty"]; color[i, j] = 0
            return obj, color"""

code = apply_action_pattern.sub(new_apply_action, code)


with open('generator/generator_interface.py', 'w') as f:
    f.write(code)

