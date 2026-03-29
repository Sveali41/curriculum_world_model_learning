import os
import sys
import torch
import numpy as np
import csv
import hydra
import traceback
from omegaconf import DictConfig, open_dict
import torch.nn.functional as F

# Fix Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Project Module Imports
from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.world_model import AttentionWM_training
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
from modelBased.policy_training.PPO import PPO
from modelBased.common.support import Support
from domain.minigrid import minigrid_support as minigrid_utils
from trainer.common.utils import (
    set_seed, validate_on_target_task
)

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
        # Dynamic action dim based on domain (Crafter=17, MiniGrid=7)
        self.action_dim = int(cfg.attention_model.action_norm_values)
        
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
            if feat.ndim == 3: feat = feat.mean(dim=1) # Spatial Pooling
            
            # Action to One-hot
            if not isinstance(action_idx, torch.Tensor):
                action_idx = torch.tensor([action_idx], device=self.device)
            else:
                action_idx = action_idx.to(self.device)
            action_oh = F.one_hot(action_idx.long(), num_classes=self.action_dim).float().to(self.device)
            
            inputs = torch.cat([feat, action_oh], dim=-1)
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
        
        action_oh = F.one_hot(action_idx.long(), num_classes=self.action_dim).float().to(self.device)
        inputs = torch.cat([feat.detach(), action_oh], dim=-1)
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
    def __init__(self, ppo, wm, device):
        self.ppo = ppo
        self.wm = wm
        self.device = device
        self.current_step_context = None
        self.expects_raw_obs = True

    def _prepare_wm_obs(self, obs_image):
        """
        Convert a raw observation into the masked WM input expected by AttentionWM.
        Crafter and MiniGrid both train the WM on local patches centered on the agent.
        """
        wm_device = next(self.wm.parameters()).device
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
        """Image (pixel) -> WM.encode -> PPO.select_action"""
        with torch.no_grad():
            obs_t = self._prepare_wm_obs(obs_image)

            # 1. BRAIN: Encode to Latent Space
            feat = self.wm.encode(obs_t)  # (1, N, D)
            feat_vec = feat.mean(dim=1)  # Spatial Pooling -> (1, 256)
            
            # 2. LIMBS: PPO Actor decision
            # PPO.select_action returns (action_idx, state, action, logprob, state_val)
            action_idx, state, action, logprob, state_val = self.ppo.select_action(feat_vec)
            
            # Store context for saving into PPO buffer when reward is ready
            self.current_step_context = (state, action, logprob, state_val)
            
            return action_idx

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
    # --------------------------------------
    # A. Setup & Initialization
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = cfg.domain
    d_cfg = cfg.domains[domain]

    # Keep domain-dependent model configs aligned with selected domain.
    with open_dict(cfg):
        cfg.attention_model.env_type = domain
        cfg.attention_model.grid_shape = d_cfg.grid_shape
        cfg.attention_model.obs_norm_values = d_cfg.obs_norm
        cfg.attention_model.action_norm_values = d_cfg.action_norm
    
    print(f"\n[P2E Baseline] Domain: {domain.upper()} | Seed: {seed}")
    transitions_per_collection = int(cfg.p2e.transitions_per_collection)
    updates_per_target = int(cfg.p2e.updates_per_target)
    target_order = str(getattr(cfg.p2e, "target_order", "sequential")).lower()
    
    # Paths & Logger
    summary_csv_path = (
        f"trainer/logs/p2e_baseline_{domain}_n{transitions_per_collection}_m{updates_per_target}_summary.csv"
    )
    os.makedirs("trainer/logs", exist_ok=True)
    summary_header = [
        "Seed",
        "Iter",
        "P2E_Mean_Reward",
        "P2E_Ensemble_Loss",
        "target_val_val_inv_loss",
        "target_val_val_ce_loss",
        "target_val_avg_val_loss_wm",
        "New_Data_Size",
        "Buffer_Size",
        "Cumulative_Transitions",
        "TargetIdx",
        "CycleIdx",
        "TargetName",
    ]
    with open(summary_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)

    # Initialize Models
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)
    p2e_ensemble = P2E_Ensemble(cfg, num_models=cfg.p2e.num_models).to(device)
    
    ppo_model = PPO(
        state_dim=cfg.attention_model.embed_dim, 
        action_dim=cfg.attention_model.action_norm_values,
        lr_actor=cfg.PPO.lr_actor,
        lr_critic=cfg.PPO.lr_critic,
        gamma=0.99, K_epochs=4, eps_clip=0.2,
        has_continuous_action_space=False
    )
    
    # ------------------------------------------------------------
    # IMPORTANT: Connect PPO to WM via Wrapper
    # ------------------------------------------------------------
    ppo_explorer = P2E_Explorer_Policy(ppo_model, wm_instance, device)
    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    support = Support(cfg)

    # Target Tasks
    if domain == "crafter":
        target_tasks = [f"crafter_target_task_{i}.txt" for i in range(1, 7)]
        target_data_dir = TRAINER_PATH.parent / "modelBased" / "data" / "train_world_model"
        val_suffix = "_uniform.npz"
    else:
        target_tasks = [f"target_task{i}.txt" for i in range(20)]
        target_data_dir = TRAINER_PATH / "data"
        val_suffix = "_test_uniform.npz"

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

    if target_order == "random":
        ordered_targets = list(np.random.permutation(target_tasks))
    else:
        ordered_targets = list(target_tasks)

    from modelBased.data.data_collect import run_env

    for target_idx, target_task in enumerate(ordered_targets):
        print(f"\n>>> P2E Target {target_idx + 1}/{len(ordered_targets)}: {target_task}")

        task_dir = (
            TRAINER_PATH / "level" / "crafter" / "target_tasks"
            if domain == "crafter"
            else TRAINER_PATH / "level" / "target_task"
        )
        task_path = task_dir / target_task

        if not task_path.exists():
            print(f"  [Error] Missing target file: {task_path}")
            continue

        try:
            env = support.wrap_env_from_text(str(task_path))
        except Exception as e:
            print(f"  [Error] Env loading failed: {e}")
            continue

        def intrinsic_reward_fn(obs, act, obs_next):
            with torch.no_grad():
                obs_t = ppo_explorer._prepare_wm_obs(obs)
                feat = wm_instance.encode(obs_t)
                r_int = p2e_ensemble.get_intrinsic_reward(feat, act)
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
            print(f"  [Data] Collected {batch_transitions} transitions, cumulative={cumulative_transitions}")

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

            print("  [Training] Updating Models...")
            if len(ppo_model.buffer.rewards) > 0:
                ppo_model.update()
            else:
                print("  [PPO] Skip update: empty rollout buffer.")

            wm_instance.train()
            replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None
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
                act_v = torch.LongTensor(act_n.flatten()).to(device)
                e_loss = p2e_ensemble.train_step(feat_all, act_v, next_feat_all)
                print(f"  [P2E] Ensemble Loss: {e_loss:.5f}")
            except Exception as e:
                print(f"  [Error] Ensemble fail: {e}")
                e_loss = 0.0

            fisher_buffer.add_from_batch(
                new_batch,
                current_sample_ratio=cfg.attention_model.ewc_ratio,
            )

            print("  [Validating] Zero-shot Multi-task Evaluation...")
            v_ls = []
            v_inv_ls = []
            v_ce_ls = []
            for t_f in ordered_targets:
                t_nn = t_f.replace(".txt", "")
                res = validate_on_target_task(
                    cfg,
                    wm_instance,
                    None,
                    str(target_data_dir),
                    f"{t_nn}{val_suffix}",
                    phase_name=f"p2e_t{target_idx + 1}_c{cycle_idx + 1}",
                    VALID_TIMES=1,
                )
                if res:
                    v_ls.append(res["avg_val_loss_wm"])
                    v_inv_ls.append(res.get("inventory_loss", np.nan))
                    v_ce_ls.append(res.get("terrain_loss", np.nan))
            if v_ls:
                m_v = float(np.mean(v_ls))
                m_inv_v = float(np.nanmean(v_inv_ls)) if len(v_inv_ls) > 0 else np.nan
                m_ce_v = float(np.nanmean(v_ce_ls)) if len(v_ce_ls) > 0 else np.nan
                print(f"  [Metrics] Mean Val Loss: {m_v:.4f}")
                with open(summary_csv_path, mode='a', newline='') as f:
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
