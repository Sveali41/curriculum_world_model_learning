import os
import sys
import torch
import numpy as np
import csv
import hydra
import traceback
from omegaconf import DictConfig, open_dict
import torch.nn.functional as F
from pathlib import Path

# Fix Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Project Module Imports
from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.common.artifacts import align_world_model_artifact_path
from modelBased.policy_training.PPO import PPO
from modelBased.common.support import Support
from domain.minigrid import minigrid_support as minigrid_utils
from domain.minigrid.minigrid_support import ColRowCanl_to_CanlRowCol
from trainer.common.utils import (
    set_seed, validate_on_all_targets
)
from trainer.common.paths import RESULTS_ROOT, VISUALIZATIONS_ROOT
from modelBased.data.data_collect import visualize_agent_coverage
from domain.minigrid.action_codec import MODEL_ACTION_COUNT

# ==============================================================================
# 1. P2E Core: Independent Ensemble Predictor Module
# ==============================================================================
class P2E_Ensemble(torch.nn.Module):
    def __init__(self, cfg, num_models=3):
        """
        Ensemble of MLP Predictors to estimate epistemic uncertainty (disagreement).
        """
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.attention_model.embed_dim
        self.action_dim = MODEL_ACTION_COUNT if cfg.domain == "minigrid" else int(cfg.attention_model.action_norm_values)
        self.is_continuous = (cfg.domain == "bipedalwalker")
        
        # Multiple MLP heads with different initializations
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embed_dim + self.action_dim, self.embed_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.embed_dim * 2, self.embed_dim)
            ) for _ in range(num_models)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(cfg.p2e.disag_lr))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def get_intrinsic_reward(self, feat, action_idx):
        """
        Compute Disagreement (Stand-dev) as intrinsic reward.
        feat: (B, N, D) -> Latent representation
        """
        with torch.no_grad():
            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(feat, device=self.device)
            else:
                feat = feat.to(self.device)
            
            # Ensure feat is (B, D)
            if feat.ndim == 3: 
                feat = feat.mean(dim=1) 
            elif feat.ndim == 1:
                feat = feat.unsqueeze(0)
            
            # Action processing
            if not isinstance(action_idx, torch.Tensor):
                action_idx = torch.as_tensor(action_idx, device=self.device)
            else:
                action_idx = action_idx.to(self.device)
            
            if self.is_continuous:
                # For Bipedal, action is already a vector (e.g. 4-dim)
                action_in = action_idx.float()
                if action_in.ndim == 1:
                    action_in = action_in.unsqueeze(0)
            else:
                # For Crafter/Minigrid, use One-hot
                if action_idx.ndim == 0 or (action_idx.ndim == 1 and action_idx.shape[0] == 1):
                    action_idx = action_idx.view(-1)
                action_in = F.one_hot(action_idx.long(), num_classes=self.action_dim).float()
            
            # Match batch size if necessary
            if action_in.shape[0] != feat.shape[0]:
                action_in = action_in.expand(feat.shape[0], -1)

            inputs = torch.cat([feat, action_in], dim=-1)
            preds = torch.stack([head(inputs) for head in self.heads]) # (K, B, D)
            
            # Disagreement (Stdev) across models
            disag = torch.std(preds, dim=0).mean(dim=-1) # (B,)
            return disag.item()

    def train_step(self, feat, action_idx, next_feat):
        """Update predictors to learn state transitions."""
        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat, device=self.device)
        else:
            feat = feat.to(self.device)
        if not isinstance(next_feat, torch.Tensor):
            next_feat = torch.as_tensor(next_feat, device=self.device)
        else:
            next_feat = next_feat.to(self.device)
        if not isinstance(action_idx, torch.Tensor):
            action_idx = torch.as_tensor(action_idx, device=self.device)
        else:
            action_idx = action_idx.to(self.device)
        
        if feat.ndim == 3: feat = feat.mean(dim=1)
        if next_feat.ndim == 3: next_feat = next_feat.mean(dim=1)
        
        if self.is_continuous:
            action_in = action_idx.float()
            if action_in.ndim == 1:
                action_in = action_in.unsqueeze(0)
        else:
            action_in = F.one_hot(action_idx.long(), num_classes=self.action_dim).float()
            
        inputs = torch.cat([feat.detach(), action_in], dim=-1)
        targets = next_feat.detach()
        
        loss = 0
        for head in self.heads:
            loss += F.mse_loss(head(inputs), targets)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item() / len(self.heads)

# ------------------------------------------------------------------------------
# 2. P2E Explorer Wrapper (Connects Brain to Limbs)
# ------------------------------------------------------------------------------
class P2E_Explorer_Policy:
    def __init__(self, ppo, wm, device, domain):
        self.ppo = ppo
        self.wm = wm
        self.device = device
        self.domain = domain
        self.current_step_context = None
        self.expects_raw_obs = True
        self.base_policy = None
        self.last_model_action_idx = None

    def set_base_policy(self, base_policy):
        self.base_policy = base_policy

    def _map_policy_action_to_env(self, action_idx):
        if self.domain != "minigrid":
            return action_idx
        # ``run_env`` owns compact->native conversion for every MiniGrid
        # policy. Returning native IDs here would convert the action twice.
        return int(action_idx)

    def _prepare_wm_obs(self, obs_image):
        """
        Convert a raw observation into the masked WM input expected by AttentionWM.
        Crafter and MiniGrid both train the WM on local patches centered on the agent.
        """
        wm_device = next(self.wm.parameters()).device

        # 1. Handle BipedalWalker (Vector observation)
        if getattr(self.wm, "is_bipedal", False):
            if isinstance(obs_image, torch.Tensor):
                obs_t = obs_image.to(wm_device).float()
            else:
                obs_t = torch.as_tensor(obs_image, device=wm_device, dtype=torch.float32)
            
            # Ensure shape is (Batch=1, Seq=1, Dim=24) for AttentionWM
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.ndim == 2:
                obs_t = obs_t.unsqueeze(1)
            return obs_t

        # 2. Handle Grid Environments (Crafter/Minigrid)
        if isinstance(obs_image, torch.Tensor):
            obs_t = obs_image.to(wm_device)
        else:
            obs_np = np.asarray(obs_image)
            if obs_np.ndim == 3:
                obs_np = np.transpose(obs_np, (2, 0, 1))
            elif obs_np.ndim == 4 and obs_np.shape[-1] <= 8:
                obs_np = np.transpose(obs_np, (0, 3, 1, 2))
            obs_t = torch.as_tensor(obs_np, device=wm_device)

        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)

        player_id = 13 if getattr(self.wm, "env_type", "") == "crafter" else 10
        agent_pos = minigrid_utils.get_agent_position(obs_t, player_id=player_id)
        masked = minigrid_utils.extract_masked_state(obs_t, self.wm.mask_size, agent_pos)
        masked = masked.to(wm_device)

        if getattr(self.wm.model, "data_type", None) != "discrete":
            masked = masked.float() / 255.0
        else:
            masked = masked.long()

        return masked

    def select_action(self, obs_image):
        """Image (pixel) -> WM.encode -> PPO.select_action + BasePolicy"""
        with torch.no_grad():
            obs_t = self._prepare_wm_obs(obs_image)

            # 1. BRAIN: Encode to Latent Space
            feat = self.wm.encode(obs_t)  # (1, N, D)
            feat_vec = feat.mean(dim=1)  # Spatial Pooling -> (1, 256)
            
            # 2. LIMBS: PPO Actor decision (Residual)
            # PPO.select_action returns (action_idx, state, action, logprob, state_val)
            action_idx, state, action, logprob, state_val = self.ppo.select_action(feat_vec)
            self.last_model_action_idx = int(action_idx) if self.domain == "minigrid" else action_idx
            
            # Store context for saving into PPO buffer when reward is ready
            self.current_step_context = (state, action, logprob, state_val)
            
            if self.base_policy is not None:
                # add_noise=True applies the 0.3 noise from YAML
                base_act = self.base_policy.select_action(obs_image, add_noise=True)
                
                if isinstance(action_idx, torch.Tensor):
                    residual_act = action_idx.cpu().numpy()
                else:
                    residual_act = np.array(action_idx)
                
                final_act = base_act + residual_act
                # Clip to env bounds (Bipedal is typically [-1, 1])
                final_act = np.clip(final_act, -1.0, 1.0)
                return final_act

            return self._map_policy_action_to_env(action_idx)

    def record_transition(self, reward, is_terminal):
        """Called by env runner after reward is available."""
        if self.current_step_context is None:
            return
        state, action, logprob, state_val = self.current_step_context
        self.ppo.save_buffer(
            state=state,
            action=action,
            logprob=logprob,
            state_value=state_val,
            reward=float(reward),
            is_terminal=bool(is_terminal),
        )
        self.current_step_context = None

# ==============================================================================
# 3. P2E Baseline Training Logic
# ==============================================================================
@hydra.main(version_base=None, config_path="conf", config_name="config_p2e")
def p2e_baseline_experiment(cfg: DictConfig):
    def _save_p2e_coverage(obs_batch, phase_name: str):
        if domain == "bipedalwalker" or obs_batch is None or len(obs_batch) == 0:
            return
        if not bool(getattr(cfg.env.collect, "save_coverage_visualize", False)):
            return
        coverage_dir = VISUALIZATIONS_ROOT / "datasets" / "p2e" / domain
        coverage_dir.mkdir(parents=True, exist_ok=True)
        save_path = coverage_dir / f"{phase_name}_coverage.png"
        try:
            visualize_agent_coverage(
                {"a": obs_batch},
                save_path=str(save_path),
                title=f"P2E Coverage ({phase_name})",
            )
            print(f"  [Coverage] Saved {save_path}")
        except Exception as e:
            print(f"  [Warn] Coverage visualization failed: {e}")

    def _ensure_summary_csv_schema(csv_path: str, expected_header: list[str]) -> bool:
        if not os.path.exists(csv_path):
            return False
        if os.path.getsize(csv_path) == 0:
            return False
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        expected_lower = ",".join([str(x).lower() for x in expected_header])
        if first_line.lower() == expected_lower:
            return True
        # Legacy schema exists: back it up and start a fresh CSV.
        backup_path = csv_path + ".legacy_backup"
        os.replace(csv_path, backup_path)
        print(f"[CSV] Existing summary with mismatched schema moved to: {backup_path}")
        return False

    # --------------------------------------
    # A. Setup & Initialization
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = cfg.domain
    is_bipedal = (domain == "bipedalwalker")
    d_cfg = cfg.domains[domain]

    # Keep domain-dependent model configs aligned with selected domain.
    with open_dict(cfg):
        cfg.attention_model.env_type = domain
        cfg.attention_model.grid_shape = d_cfg.grid_shape
        cfg.attention_model.obs_norm_values = d_cfg.obs_norm
        cfg.attention_model.action_norm_values = d_cfg.action_norm
        cfg.attention_model.validation_metric = d_cfg.validation_metric
        cfg.attention_model.data_type = d_cfg.data_type
    align_world_model_artifact_path(cfg)
    
    print(f"\n[P2E Baseline] Domain: {domain.upper()} | Seed: {seed}")
    default_n = int(cfg.p2e.transitions_per_collection)
    default_m = int(cfg.p2e.updates_per_target)
    per_domain_cfg = getattr(cfg.p2e, "per_domain", None)
    domain_cfg = None
    if per_domain_cfg is not None:
        domain_cfg = per_domain_cfg.get(domain, None)

    transitions_per_collection = int(
        getattr(domain_cfg, "transitions_per_collection", default_n)
        if domain_cfg is not None else default_n
    )
    updates_per_target = int(
        getattr(domain_cfg, "updates_per_target", default_m)
        if domain_cfg is not None else default_m
    )
    target_order = str(getattr(cfg.p2e, "target_order", "sequential")).lower()
    mask_suffix = f"_mask{int(getattr(cfg.attention_model, 'attention_mask_size', 0))}"
    
    # Paths & Logger
    log_dir = Path(getattr(cfg, "p2e_log_dir", RESULTS_ROOT / "p2e"))
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = str(
        log_dir
        / f"p2e_baseline_{domain}{mask_suffix}_n{transitions_per_collection}_m{updates_per_target}_summary.csv"
    )
    if domain == "minigrid":
        summary_header = [
            "seed", "Iter", "P2E_Mean_Reward", "P2E_Ensemble_Loss",
            "mode", "phase", "transitions", "avg_target_loss"
        ]
        file_non_empty = _ensure_summary_csv_schema(summary_csv_path, summary_header)
    elif is_bipedal:
        summary_header = [
            "Seed", "Iter", "P2E_Mean_Reward", "P2E_Ensemble_Loss",
            "target_val_contact_acc", "target_val_contact_bce", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Cumulative_Transitions", "TargetIdx", "CycleIdx", "TargetName"
        ]
        file_non_empty = os.path.exists(summary_csv_path) and os.path.getsize(summary_csv_path) > 0
    else:
        summary_header = [
            "Seed", "Iter", "P2E_Mean_Reward", "P2E_Ensemble_Loss",
            "target_val_val_inv_loss", "target_val_val_ce_loss", "target_val_avg_val_loss_wm",
            "New_Data_Size", "Buffer_Size", "Cumulative_Transitions", "TargetIdx", "CycleIdx", "TargetName"
        ]
        file_non_empty = os.path.exists(summary_csv_path) and os.path.getsize(summary_csv_path) > 0

    if not file_non_empty:
        with open(summary_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(summary_header)

    # Initialize Models
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)
    p2e_ensemble = P2E_Ensemble(cfg, num_models=cfg.p2e.num_models).to(device)
    
    ppo_action_dim = MODEL_ACTION_COUNT if domain == "minigrid" else cfg.attention_model.action_norm_values
    ppo_model = PPO(
        state_dim=cfg.attention_model.embed_dim, 
        action_dim=ppo_action_dim,
        lr_actor=cfg.PPO.lr_actor,
        lr_critic=cfg.PPO.lr_critic,
        gamma=0.99, K_epochs=4, eps_clip=0.2,
        has_continuous_action_space=is_bipedal
    )
    
    # ------------------------------------------------------------
    # IMPORTANT: Connect PPO to WM via Wrapper
    # ------------------------------------------------------------
    ppo_explorer = P2E_Explorer_Policy(ppo_model, wm_instance, device, domain)
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    support = Support(cfg)

    # Target Tasks
    if domain == "crafter":
        target_tasks = [f"crafter_target_task_{i}.txt" for i in range(1, 21)]
        target_data_dir = Path(str(getattr(d_cfg, "target_tasks_folder", TRAINER_PATH / "data" / "crafter" / "target_tasks")))
        val_suffix = str(getattr(d_cfg, "target_task_suffix", "_uniform.npz"))
    elif domain == "bipedalwalker":
        target_tasks = [f"bipedal_target_task_{i}.txt" for i in range(1, 21)]
        target_data_dir = Path(str(getattr(d_cfg, "target_tasks_folder", TRAINER_PATH / "data" / "bipedalwalker" / "target_tasks")))
        val_suffix = str(getattr(d_cfg, "target_task_suffix", "_uniform.npz"))
    else:
        target_tasks = [f"target_task{i}.txt" for i in range(20)]
        target_data_dir = Path(str(getattr(d_cfg, "target_tasks_folder", TRAINER_PATH / "data" / "minigrid" / "target_tasks")))
        val_suffix = str(getattr(d_cfg, "target_task_suffix", "_test_uniform.npz"))

    total_target_budget = transitions_per_collection * updates_per_target
    total_budget = total_target_budget * len(target_tasks)
    print(
        f"[P2E Budget] n={transitions_per_collection}, m={updates_per_target}, "
        f"y={len(target_tasks)} => per-target={total_target_budget}, total={total_budget}"
    )

    # --------------------------------------
    # B. Main Loop
    # --------------------------------------
    old_params, fisher = None, None
    cumulative_transitions = 0
    fisher_target_shape = None  # (H, W) used to keep replay samples shape-consistent for MiniGrid

    if target_order == "random":
        ordered_targets = list(np.random.permutation(target_tasks))
    else:
        ordered_targets = list(target_tasks)

    from modelBased.data.data_collect import run_env

    for target_idx, target_task in enumerate(ordered_targets):
        print(f"\n>>> P2E Target {target_idx + 1}/{len(ordered_targets)}: {target_task}")

        if domain == "crafter":
            task_dir = TRAINER_PATH / "level" / "crafter" / "target_tasks"
        elif domain == "bipedalwalker":
            task_dir = TRAINER_PATH / "level" / "bipedal_walker" / "target_tasks"
        else:
            task_dir = TRAINER_PATH / "level" / "minigrid" / "target_task"
        task_path = task_dir / target_task

        if not task_path.exists():
            print(f"  [Error] Missing target file: {task_path}")
            continue

        try:
            env = support.wrap_env_from_text(str(task_path))
        except Exception as e:
            print(f"  [Error] Env loading failed: {e}")
            continue

        from omegaconf import OmegaConf
        with open_dict(cfg):
            if not hasattr(cfg.env, "collect"):
                cfg.env.collect = OmegaConf.create({})
            # Ensure exploration mode is random (physical) instead of uniform (teleport) for ALL domains
            cfg.env.collect.data_type = "random"

        if domain == "bipedalwalker":
            from modelBased.data.data_collect import _build_bipedal_behavior_policy
            base_policy = _build_bipedal_behavior_policy(cfg, env)
            ppo_explorer.set_base_policy(base_policy)
        elif domain == "crafter":
            # Grid environments use pure PPO exploration
            ppo_explorer.set_base_policy(None)
        elif domain == "minigrid":
            # Grid environments use pure PPO exploration
            ppo_explorer.set_base_policy(None)
        else:
            ppo_explorer.set_base_policy(None)

        def intrinsic_reward_fn(obs, act, obs_next):
            with torch.no_grad():
                obs_t = ppo_explorer._prepare_wm_obs(obs)
                feat = wm_instance.encode(obs_t)
                model_act = ppo_explorer.last_model_action_idx if domain == "minigrid" else act
                r_int = p2e_ensemble.get_intrinsic_reward(feat, model_act)
                return float(r_int) * cfg.p2e.intrinsic_reward_scale

        for cycle_idx in range(updates_per_target):
            print(
                f"  [Exploration] Target {target_idx + 1}/{len(ordered_targets)} "
                f"Cycle {cycle_idx + 1}/{updates_per_target} | Collect {transitions_per_collection} transitions"
            )

            wm_instance.eval()
            p2e_ensemble.eval()

            # Bound collection by transition budget instead of episode count.
            with open_dict(cfg):
                cfg.env.collect.maximum_dataset_size = transitions_per_collection
                cfg.env.collect.mini_dataset_size = transitions_per_collection
                cfg.env.collect.episodes = transitions_per_collection

            try:
                obs_n, obsn_n, act_n, rew_n, done_n, info_n, inv_n, inv_next_n = run_env(
                    env,
                    cfg,
                    wandb_run=None,
                    log_name="p2e",
                    policy=ppo_explorer,
                    intrinsic_reward_fn=intrinsic_reward_fn,
                )
            except Exception as e:
                print(f"  [Error] Rollout failed: {e}")
                traceback.print_exc()
                env.close()
                continue

            batch_transitions = int(len(obs_n))
            cumulative_transitions += batch_transitions

            # [ROOT CAUSE FIX] Use the exact same transpose logic as Target Baseline (save_experiments)
            # This ensures (N, W, H, C) -> (N, C, H, W) where spatial axes are swapped correctly.
            if domain == "minigrid" and obs_n.ndim == 4:
                obs_n = ColRowCanl_to_CanlRowCol(obs_n)
                obsn_n = ColRowCanl_to_CanlRowCol(obsn_n)

            print(f"  [Data] Collected {batch_transitions} transitions, cumulative={cumulative_transitions}")
            _save_p2e_coverage(
                obs_n,
                phase_name=f"p2e_t{target_idx + 1}_c{cycle_idx + 1}",
            )

            if domain == "minigrid" and isinstance(obs_n, np.ndarray) and obs_n.ndim == 4:
                # Robustly detect spatial dimensions (Width, Height)
                if obs_n.shape[1] <= 8:  # Likely NCHW
                    current_shape = (int(obs_n.shape[2]), int(obs_n.shape[3]))
                else:  # Likely NHWC
                    current_shape = (int(obs_n.shape[1]), int(obs_n.shape[2]))

                if fisher_target_shape is None:
                    fisher_target_shape = current_shape
                else:
                    grown_shape = (
                        max(fisher_target_shape[0], current_shape[0]),
                        max(fisher_target_shape[1], current_shape[1]),
                    )
                    if grown_shape != fisher_target_shape:
                        # Existing buffer items may be in smaller shapes.
                        # Harmonize them to the new target shape instead of clearing replay.
                        print(
                            f"  [Replay] Map shape grew from {fisher_target_shape} to {grown_shape}; "
                            f"harmonizing existing Fisher buffer samples."
                        )
                        fisher_target_shape = grown_shape
                        changed = fisher_buffer.harmonize_buffer_map_shape(fisher_target_shape)
                        if changed > 0:
                            print(f"  [Replay] Harmonized {changed} replay sample fields to shape {fisher_target_shape}.")

            new_batch = {
                "a": obs_n,
                "b": obsn_n,
                "c": act_n,
                "d": rew_n,
                "e": done_n,
                "f": info_n,
                "g": inv_n,
                "h": inv_next_n,
                "obs": obs_n,
                "obs_next": obsn_n,
                "act": act_n,
                "reward": rew_n,
                "done": done_n,
                "info": info_n,
                "inv": inv_n,
                "inv_next": inv_next_n,
            }

            # Save the collected data to `.npz` for later auditing.
            try:
                save_dir = TRAINER_PATH / "data" / domain / "target_tasks"
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"p2e_{target_task}_c{cycle_idx+1}.npz"
                np.savez(save_dir / save_name, **new_batch)
                print(f"  [Save] P2E data saved to: {save_dir / save_name}")
            except Exception as e:
                print(f"  [Error] Failed to save P2E .npz: {e}")

            print("  [Training] Updating Models...")
            if len(ppo_model.buffer.rewards) > 0:
                ppo_model.update()
            else:
                print("  [PPO] Skip update: empty rollout buffer.")

            wm_instance.train()
            replay_data = None
            if len(fisher_buffer) > 0:
                try:
                    replay_data = fisher_buffer.export_dict()
                except ValueError as e:
                    # Try one more shape harmonization pass before skipping replay.
                    print(f"  [Warn] Replay export skipped due to shape mismatch: {e}")
                    if domain == "minigrid" and fisher_target_shape is not None:
                        changed = fisher_buffer.harmonize_buffer_map_shape(fisher_target_shape)
                        if changed > 0:
                            print(f"  [Replay] Re-harmonized {changed} sample fields; retrying replay export.")
                            try:
                                replay_data = fisher_buffer.export_dict()
                            except ValueError as e2:
                                print(f"  [Warn] Replay export still failed after harmonization: {e2}")
                                replay_data = None
                        else:
                            replay_data = None
                    else:
                        replay_data = None
            train_res, fisher, _ = AttentionWM_training.train_api(
                cfg,
                wm_instance,
                old_params,
                fisher,
                replay_data=replay_data,
                direct_data=new_batch,
            )
            old_params = train_res.get("old_params")

            try:
                wm_instance.eval()
                with torch.no_grad():
                    obs_v = ppo_explorer._prepare_wm_obs(obs_n)
                    obsn_v = ppo_explorer._prepare_wm_obs(obsn_n)
                    feat_all = wm_instance.encode(obs_v)
                    next_feat_all = wm_instance.encode(obsn_v)
                if is_bipedal:
                    act_v = torch.as_tensor(act_n, dtype=torch.float32, device=device)
                else:
                    act_v = torch.LongTensor(act_n.flatten()).to(device)
                e_loss = p2e_ensemble.train_step(feat_all, act_v, next_feat_all)
                print(f"  [P2E] Ensemble Loss: {e_loss:.5f}")
            except Exception as e:
                print(f"  [Error] Ensemble fail: {e}")
                e_loss = 0.0

            fisher_buffer.add_from_batch(
                new_batch,
                current_sample_ratio=cfg.attention_model.current_sample_ratio,
                target_shape=fisher_target_shape if domain == "minigrid" else None,
            )

            print("  [Validating] Zero-shot Multi-task Evaluation...")
            val_summary = validate_on_all_targets(
                cfg,
                wm_instance,
                str(target_data_dir),
                ordered_targets,
                val_suffix,
                phase_name=f"p2e_t{target_idx + 1}_c{cycle_idx + 1}",
                VALID_TIMES=1,
            )
            if val_summary["valid_count"] > 0:
                m_v = float(val_summary["avg_val_loss_wm"])
                if is_bipedal:
                    m_contact_acc = float(val_summary.get("contact_acc", np.nan))
                    m_contact_bce = float(val_summary.get("contact_bce", np.nan))
                else:
                    m_inv_v = float(val_summary.get("inventory_loss", np.nan))
                    m_ce_v = float(val_summary.get("terrain_loss", np.nan))
                print(f"  [Metrics] Mean Val Loss: {m_v:.4f}")
                with open(summary_csv_path, mode='a', newline='') as f:
                    if domain == "minigrid":
                        phase_name = f"P2E_T{target_idx + 1}_C{cycle_idx + 1}"
                        iter_idx = target_idx * updates_per_target + cycle_idx + 1
                        csv.writer(f).writerow([
                            seed,
                            iter_idx,
                            float(np.mean(rew_n)) if len(rew_n) > 0 else np.nan,
                            e_loss,
                            "P2E",
                            phase_name,
                            cumulative_transitions,
                            m_v,
                        ])
                    elif is_bipedal:
                        csv.writer(f).writerow([
                            seed,
                            target_idx * updates_per_target + cycle_idx + 1,
                            float(np.mean(rew_n)) if len(rew_n) > 0 else np.nan,
                            e_loss,
                            m_contact_acc,
                            m_contact_bce,
                            m_v,
                            batch_transitions,
                            len(fisher_buffer),
                            cumulative_transitions,
                            target_idx + 1,
                            cycle_idx + 1,
                            target_task,
                        ])
                    else:
                        csv.writer(f).writerow([
                            seed,
                            target_idx * updates_per_target + cycle_idx + 1,
                            float(np.mean(rew_n)) if len(rew_n) > 0 else np.nan,
                            e_loss,
                            m_inv_v,
                            m_ce_v,
                            m_v,
                            batch_transitions,
                            len(fisher_buffer),
                            cumulative_transitions,
                            target_idx + 1,
                            cycle_idx + 1,
                            target_task,
                        ])

            torch.cuda.empty_cache()

        env.close()

    print(f">>> P2E Finished. Results: {summary_csv_path}")

if __name__ == "__main__":
    p2e_baseline_experiment()
