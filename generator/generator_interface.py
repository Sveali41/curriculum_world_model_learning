import torch
import numpy as np
import torch.nn.functional as F

from generator.generator_agent import GeneratorPPO
from generator.reward_system import DiversityModule, check_solvability, calculate_lp_reward
from generator.env_designer import PCGSeeder, task_placer
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from modelBased.common.support import Support
import copy
from modelBased.common.utils import TRAINER_PATH
from trainer.common.utils import extract_loss_map_over_validations, collect_data_general
import os




# Generator action vocabulary (8 actions)
ACTION_TABLE = {
    0: None,  # No-op
    1: ("key", "red"),
    2: ("key", "green"),
    3: ("key", "blue"),
    4: ("door", "red"),    # open door
    5: ("door", "green"),
    6: ("door", "blue"),
    7: ("lava", None),
}


class GeneratorInterface:
    def __init__(self, world_model, device, cfg):
        self.device = device
        self.cfg = cfg
        hparams = cfg.generator_agent
        self.batch_size = hparams.batch_size
        self.support = Support(cfg)
        self.wm = world_model
        self.ppo = GeneratorPPO(context_dim=hparams.context_dim,   
            num_actions=len(ACTION_TABLE),
            his_emb_dim=hparams.his_emb_dim,
            ratio=hparams.ratio
        )
        self.diversity = DiversityModule()
        self.map_height = hparams.map_height
        self.map_width = hparams.map_width
        self.max_edits = hparams.max_edits
        self.seeder = PCGSeeder(height=self.map_height, width=self.map_width)
        self.prev_data = None
        self.OBJ_START = OBJECT_TO_IDX["agent"]
        self.OBJ_GOAL = OBJECT_TO_IDX["goal"]
        self.OBJ_EMPTY = OBJECT_TO_IDX["empty"]

    # ------------------------------------------------------------
    def sync_world_model(self, state_dict):
        self.wm.load_state_dict(state_dict)

    # ------------------------------------------------------------
    def step(self):
        base_maps = []

        for _ in range(self.batch_size):
            z = np.random.randint(0, 1e6)
            grid = self.seeder.generate(z=z)
            grid, _ = task_placer(grid)
            base_maps.append(grid)

        base_ids = torch.tensor(base_maps, device=self.device) # convert to tensor
        num_classes = int(base_ids.max()) + 1 # element space size

        curr_map = (
            F.one_hot(base_ids, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        mask = self._immutable_mask(base_ids)

        # Initialize history context for the first iteration.
        # If no previous rollout information is available, use a zero context
        # as a placeholder for history-conditioned generation.
        if self.prev_data is None:
            self.prev_data = self._zero_context(curr_map.size(0), self.map_height, self.map_width)

        actions, logp, values, _ = self.ppo.select_action(
            curr_map, self.prev_data, mask, max_edits=self.max_edits

        )

        next_maps, next_heats, next_attns = [], [], []
        valid_trajs = []

        for i in range(self.batch_size):
            obj_map, color_map = self._apply_action(
                base_ids[i].cpu().numpy(),
                actions[i].cpu().numpy(),
            )
            final_map = np.stack([obj_map, color_map], axis=0)

            if not check_solvability(obj_map):
                self._save(i, -5.0, curr_map, mask, actions, logp, values)
                continue

            traj, heat, attn, solved = self._rollout_env(final_map)
            if not solved:
                self._save(i, -1.0, curr_map, mask, actions, logp, values)
                continue

            r_lp = calculate_lp_reward(self.wm, traj, self.device)
            r_div = self.diversity.get_reward(
                torch.tensor(final_map).unsqueeze(0).to(self.device)
            )

            self._save(i, r_lp + 0.1 * r_div, curr_map, mask, actions, logp, values)

            valid_trajs.append(traj)
            next_maps.append(self._map_to_tensor(final_map))
            next_heats.append(heat)
            next_attns.append(attn)

        if len(next_maps) > 0:
            self.prev_data = (
                torch.cat(next_maps),
                torch.cat(next_heats),
                torch.cat(next_attns),
            )

        return valid_trajs

    def update(self):
        return self.ppo.update()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _immutable_mask(self, ids):
        """
        Create an immutable mask for generator editing.

        Immutable positions (mask = 1.0):
        - start
        - goal
        - non-empty cells (e.g. wall)

        Mutable positions (mask = 0.0):
        - empty cells

        ids:    [B, H, W] object id map
        return: [B, 1, H, W] float tensor (1.0 = immutable)
        """
        mask = torch.ones_like(ids, dtype=torch.float32)

        # Only empty cells are editable
        mask[ids == self.OBJ_EMPTY] = 0.0

        # Explicitly protect start & goal (redundant but safe)
        mask[ids == self.OBJ_START] = 1.0
        mask[ids == self.OBJ_GOAL] = 1.0

        return mask.unsqueeze(1)


    def _apply_action(self, base_obj_map, act):
        H, W = base_obj_map.shape

        # 1) 复制 object map
        obj = base_obj_map.copy()

        # 2) 初始化 color / state（全 0 或默认）
        color = np.zeros((H, W), dtype=np.int64)
        immutable = (obj == self.OBJ_START) | (obj == self.OBJ_GOAL)

        for i in range(H):
            for j in range(W):
                if immutable[i, j]:
                    continue

                a = act[i, j]
                if a == 0:
                    continue  # No-op

                obj_type, color_name = ACTION_TABLE[a]

                if obj_type == "key":
                    obj[i, j] = OBJECT_TO_IDX["key"]
                    color[i, j] = COLOR_TO_IDX[color_name]

                elif obj_type == "door":
                    obj[i, j] = OBJECT_TO_IDX["door"]
                    color[i, j] = COLOR_TO_IDX[color_name]
                    

                elif obj_type == "lava":
                    obj[i, j] = OBJECT_TO_IDX["lava"]

        return obj, color

    def _save(self, i, r, maps, masks, acts, lps, vals):
        pm, ph, pa = self.prev_data
        self.ppo.save_buffer(
            maps[i : i + 1],
            (pm[i : i + 1], ph[i : i + 1], pa[i : i + 1]),
            masks[i : i + 1],
            acts[i : i + 1],
            lps[i : i + 1],
            vals[i : i + 1],
            r,
        )

    def _zero_context(self, B, H, W):
        return (
            torch.zeros((B, 3, H, W), device=self.device),
            torch.zeros((B, 1, H, W), device=self.device),
            torch.zeros((B, 1, H, W), device=self.device),
        )

    def _map_to_tensor(self, m):
        return torch.tensor(m, device=self.device).unsqueeze(0).repeat(1, 3, 1, 1)

    def _rollout_env(self, map_obj):
        """
        Run the environment with the generated map to collect trajectory.
        Then compute LP reward (outside) and attention/heat maps (here).
        """
        # 1. Wrap env
        obj_map, color_map = map_obj
        map_tensor = torch.tensor(obj_map, dtype=torch.long, device=self.device)
        color_tensor = torch.tensor(color_map, dtype=torch.long, device=self.device)

        try:
            obj_str, color_str = self.support.interpret_env(map_tensor.cpu(), color_array=color_tensor.cpu())
            env_str = (obj_str, color_str)
        except Exception as e:
            print(f"Error wrapping env: {e}")
            return [], None, None, False

        # 2. Setup paths and config for data reuse
        temp_data_path = os.path.join(self.cfg.env.collect.data_save_path, "generator_rollout_temp.npz")
        # Ensure directory exists
        os.makedirs(os.path.dirname(temp_data_path), exist_ok=True)

        old_path = self.support.cfg.env.collect.data_save_path
        old_episodes = self.support.cfg.env.collect.episodes
        
        self.support.cfg.env.collect.data_save_path = temp_data_path
        self.support.cfg.env.collect.episodes = 100 
        
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)

        try:
            # 3. Collect Data
            # Note: collect_data_trainer handles run_env call and saving.
            collect_data_general(
                self.support.cfg,
                env_source=env_str,
                save_name='UED_temp_data_path',
                max_steps=1000,
                maximum_dataset_size=300000,
                recollect_data=False
            )
            
            # 4. Load Data (for Traj)
            if os.path.exists(temp_data_path):
                task_npz = np.load(temp_data_path, allow_pickle=True)
                # Convert to dict format for return
                traj_data = {
                    'obs': torch.tensor(task_npz['a'], device=self.device),
                    'obs_next': torch.tensor(task_npz['b'], device=self.device),
                    'act': torch.tensor(task_npz['c'], device=self.device),
                    'info': task_npz['f'] if 'f' in task_npz else None
                }
                
                rew_np = task_npz['d']
                done_np = task_npz['e']
                solved = np.any((done_np) & (rew_np > 0))
            else:
                print("Error: Rollout data file not found.")
                return [], None, None, False

            # 5. Compute Heat Map (Reuse validation function)
            avg_loss_map, _ = extract_loss_map_over_validations(
                self.cfg,
                net=self.wm,
                old_params=None, # Use current weights
                data_dir=temp_data_path,
                valid_times=1
            )
            
            heat = torch.tensor(avg_loss_map, device=self.device).unsqueeze(0).unsqueeze(0)
            attn = torch.zeros_like(heat) # Placeholder as before

        except Exception as e:
            print(f"Error in rollout/heat computation: {e}")
            traj_data = {}
            heat = torch.zeros((1, 1, self.map_height, self.map_width), device=self.device)
            attn = torch.zeros((1, 1, self.map_height, self.map_width), device=self.device)
            solved = False
        finally:
            # Restore config
            self.support.cfg.env.collect.data_save_path = old_path
            self.support.cfg.env.collect.episodes = old_episodes
            # Optional: remove temp file
            # if os.path.exists(temp_data_path):
            #     os.remove(temp_data_path)

        return traj_data, heat, attn, solved
