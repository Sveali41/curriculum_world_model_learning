import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from generator.generator_network import MapEditorActorCritic
from generator.history_encoder import HistoryEncoder
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneratorPPO:
    """
    PPO-based Generator with a learnable History Encoder.
    """

    def __init__(
        self,
        his_emb_dim=16,
        context_dim=64,
        lr_actor=1e-4, # [TUNED] 3e-4 -> 1e-4: Calm down. Fine-tune instead of jumping.
        lr_critic=3e-4,
        gamma=0.99,
        K_epochs=10,   # Keep K_epochs=10 high to learn efficiently from safe steps
        eps_clip=0.2,
        entropy_coef=0.02,
        entropy_coef_start=None,
        entropy_coef_end=None,
        entropy_anneal_iters=0,
        buffer_window_rounds=1,
        num_actions=11,
        # ratio=0.25, # removed
        top_k_features=16,
        ablation_type="none",
        env_type="minigrid",
        spatial_dpp_sigma=1.5,

    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = int(K_epochs)
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = float(entropy_coef if entropy_coef_start is None else entropy_coef_start)
        self.entropy_coef_end = float(entropy_coef if entropy_coef_end is None else entropy_coef_end)
        self.entropy_anneal_iters = max(0, int(entropy_anneal_iters))
        self.buffer_window_rounds = max(1, int(buffer_window_rounds))
        self.context_dim = context_dim
        self.env_type = str(env_type).lower()
        # self.ratio = ratio # removed
        self.top_k_features = top_k_features
        self.spatial_dpp_sigma = float(spatial_dpp_sigma)
        self.is_bipedal = (self.env_type == "bipedalwalker")
        self.is_minigrid = (self.env_type == "minigrid")
        self.policy_context_dim = (
            context_dim
            if self.is_bipedal or self.is_minigrid
            else context_dim * 2
        )

        # history encoder
        

        # === Networks ===
        if ablation_type == "no_history":
            self.encoder = None  # No history encoder needed for this ablation
        else:
            self.encoder = HistoryEncoder(
                context_dim=context_dim,
                emb_dim=his_emb_dim,
                env_type=env_type,
            ).to(device)

        self.policy = MapEditorActorCritic(
            num_actions=num_actions,
            context_dim=self.policy_context_dim,
            ablation_type=ablation_type,
            env_type=env_type,
            spatial_dpp_sigma=spatial_dpp_sigma,
        ).to(device)

        self.policy_old = MapEditorActorCritic(
            num_actions=num_actions,
            context_dim=self.policy_context_dim,
            ablation_type=ablation_type,
            env_type=env_type,
            spatial_dpp_sigma=spatial_dpp_sigma,
        ).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # === Optimizer (encoder + policy) ===
        optim_params = []

        if self.encoder is not None:
            optim_params.append(
                {"params": self.encoder.parameters(), "lr": lr_actor}
            )

        optim_params += [
            {"params": self.policy.stem.parameters(), "lr": lr_actor},
            {"params": self.policy.res_blocks.parameters(), "lr": lr_actor},
            {"params": self.policy.emb_obj.parameters(), "lr": lr_actor},
            {"params": self.policy.emb_color.parameters(), "lr": lr_actor},
            {"params": self.policy.emb_state.parameters(), "lr": lr_actor},
            {"params": self.policy.actor.parameters(), "lr": lr_actor},
            {"params": self.policy.stats_actor.parameters(), "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ]
        if self.policy.history_fusion is not None:
            optim_params.extend([
                {"params": self.policy.history_fusion.parameters(), "lr": lr_actor},
                {"params": self.policy.history_type_actor.parameters(), "lr": lr_actor},
            ])

        self.optimizer = optim.Adam(optim_params)

        self.mse = nn.MSELoss()

        # === PPO buffer with a consistent per-sample layout ===
        self.buffer = {
            "curr_map": [],
            "prev_map": [],
            "prev_heat": [],
            "mask": [],
            "action": [],         # Terrain actions (grid)
            "logprob": [],        # Terrain logprobs (summed grid)
            "stats_action": [],   # Inventory actions (vector)
            "stats_logprob": [],  # Inventory logprobs (summed vector)
            "value": [],
            "reward": [],
            "topk_mask": [],
            "stats_topk_mask": [],
            "stats_heat": [],     # Inventory error history [1, 16]
        }
        self.last_mean_reward = 0.0
        self.last_entropy_coef = float(self.entropy_coef_start)
        self.round_lengths = deque()
        self.current_round_count = 0

    def _get_entropy_coef(self, iteration=None):
        if iteration is None or self.entropy_anneal_iters <= 0:
            return float(self.entropy_coef)
        if iteration >= self.entropy_anneal_iters:
            return float(self.entropy_coef_end)

        progress = float(iteration) / float(max(self.entropy_anneal_iters, 1))
        return float(
            self.entropy_coef_start
            + (self.entropy_coef_end - self.entropy_coef_start) * progress
        )

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def _compute_global_context_dual(self, prev_map, terrain_heat, stats_heat, top_k_features=None):
        """
        Aggregate spatial failure features and inventory failure features.
        Crafter keeps both:
        - local per-sample history context
        - global batch summary context

        MiniGrid keeps only its per-sample semantic context. Broadcasting a
        batch-wise max makes every generated map chase the same dominant
        pattern and is unnecessary once absolute coordinates are removed.
        """
        top_k_features = self.top_k_features if top_k_features is None else int(top_k_features)
        # Use the HistoryEncoder to extract per-sample failure features.
        ctx = self.encoder(prev_map, terrain_heat, stats_heat) # [B, context_dim]

        if self.is_bipedal or self.is_minigrid:
            return F.normalize(ctx, p=2, dim=1)
        
        local_ctx = F.normalize(ctx, p=2, dim=1)

        # Aggregate across the batch with max pooling.
        v_ctx, _ = torch.max(ctx, dim=0, keepdim=True) # [1, context_dim]

        # Keep only the most salient features via top-k sparsification.
        if v_ctx.size(1) > top_k_features:
            top_val, _ = torch.topk(v_ctx, k=top_k_features, dim=1)
            min_val = top_val[:, -1:]
            mask = (v_ctx >= min_val).float()
            v_ctx = v_ctx * mask

        # Normalize feature magnitudes.
        global_ctx = F.normalize(v_ctx, p=2, dim=1)
        global_ctx = global_ctx.expand(local_ctx.size(0), -1)
        return torch.cat([local_ctx, global_ctx], dim=1)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, base_map, prev_data, mask, max_edits_layout, max_edits_stats):
        B = base_map.size(0)
        if self.encoder is None:   
            ctx = None
            global_ctx = None
            phs = None
        else:
            if prev_data is None:
                # Initial state: empty map, empty terrain heatmap, empty inventory heatmap.
                H, W = base_map.size(2), base_map.size(3)
                # pm: prev_map, pht: prev_heat_terrain, phs: prev_heat_stats
                pm = torch.zeros((1, 3, H, W), device=device)
                feedback_channels = 2 if self.env_type == "minigrid" else 1
                pht = torch.zeros((1, feedback_channels, H, W), device=device)
                phs = None if self.env_type == "minigrid" else torch.zeros(
                    (1, 26 if self.env_type == "bipedalwalker" else 16), device=device
                )
                global_ctx = self._compute_global_context_dual(pm, pht, phs)
            else:
                if len(prev_data) == 2:
                    pm, pht = prev_data
                    phs = None
                else:
                    pm, pht, phs = prev_data
                global_ctx = self._compute_global_context_dual(pm, pht, phs)
            if global_ctx.size(0) == B:
                ctx = global_ctx
            else:
                ctx = global_ctx.repeat(B, 1)
            if phs is not None and phs.size(0) == 1 and B > 1:
                phs = phs.repeat(B, 1)

        action, stats_act, map_logp, stats_logp, value, topk_mask, topk_stats_mask = self.policy_old.act(
            base_map, ctx, mask, max_edits_layout, max_stats_edit_ratio=max_edits_stats, stats_heat=phs
        )

        # map_logp is [B, H, W], stats_logp is [B] (already summed in network)
        total_logprob = map_logp.sum(dim=(1, 2)) + stats_logp
        return action, stats_act, total_logprob, value, topk_mask, topk_stats_mask, global_ctx

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------
    def save_buffer(self, curr_map, prev_data, mask, action, stats_action, logprob, value, reward, topk_mask, stats_topk_mask):
        self.buffer["curr_map"].append(curr_map.cpu())
        self.buffer["mask"].append(mask.cpu())
        self.buffer["action"].append(action.cpu())
        self.buffer["stats_action"].append(stats_action.cpu())
        self.buffer["logprob"].append(logprob.cpu())
        self.buffer["value"].append(value.cpu())
        self.buffer["reward"].append(float(reward))
        self.buffer["topk_mask"].append(topk_mask.cpu())
        self.buffer["stats_topk_mask"].append(stats_topk_mask.cpu())

        if prev_data is None:
            B, _, H, W = curr_map.shape
            self.buffer["prev_map"].append(torch.zeros((B, 3, H, W)))
            feedback_channels = 2 if self.env_type == "minigrid" else 1
            self.buffer["prev_heat"].append(
                torch.zeros((B, feedback_channels, H, W))
            )
            stats_dim = 26 if self.env_type == "bipedalwalker" else 16
            self.buffer["stats_heat"].append(torch.zeros((B, stats_dim)))
        else:
            if len(prev_data) == 2:
                pm, pht = prev_data
                phs = None
            else:
                pm, pht, phs = prev_data
            self.buffer["prev_map"].append(pm.cpu())
            # We rename prev_heat internally to match dual streams
            self.buffer["prev_heat"].append(pht.cpu())
            if phs is None:
                B = pm.size(0)
                stats_dim = 26 if self.env_type == "bipedalwalker" else 16
                self.buffer["stats_heat"].append(torch.zeros((B, stats_dim)))
            else:
                self.buffer["stats_heat"].append(phs.cpu())
        self.current_round_count += 1

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k].clear()
        self.round_lengths.clear()
        self.current_round_count = 0

    def _trim_buffer_to_recent_rounds(self):
        while len(self.round_lengths) > self.buffer_window_rounds:
            remove_n = int(self.round_lengths.popleft())
            if remove_n <= 0:
                continue
            for k in self.buffer:
                if remove_n >= len(self.buffer[k]):
                    self.buffer[k].clear()
                else:
                    del self.buffer[k][:remove_n]

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self, iteration=None):
            """
            PPO update logic:
            1. Extract and process rollout data from the buffer.
            2. Recompute context features for the HistoryEncoder.
            3. Compute clipped PPO loss, value loss, and entropy loss.
            4. Backpropagate and synchronize the old policy.
            """
            # --- Step 0: Safety checks ---
            if len(self.buffer["curr_map"]) == 0:
                print("[GeneratorPPO] Warning: Buffer is empty, skipping update.")
                return 0.0, 0.0
            if self.current_round_count > 0:
                self.round_lengths.append(self.current_round_count)
                self.current_round_count = 0
            # PPO is on-policy. Drop rounds outside the configured window
            # before computing this update, rather than training on one extra
            # stale round and trimming it only afterward.
            self._trim_buffer_to_recent_rounds()

            # --- Step 1: Basic data preparation and normalization ---
            rewards = torch.tensor(self.buffer["reward"], device=device)

            # Monitor the mean reward as a coarse convergence signal.
            mean_reward = rewards.mean().item()
            self.last_mean_reward = mean_reward
            print(f"[GeneratorPPO] Mean Reward: {mean_reward:.4f}")
            current_entropy_coef = self._get_entropy_coef(iteration)
            self.last_entropy_coef = current_entropy_coef
            print(f"[GeneratorPPO] Entropy Coef: {current_entropy_coef:.4f}")
            print(
                f"[GeneratorPPO] Buffer Window: {len(self.round_lengths)} round(s), "
                f"{len(self.buffer['reward'])} samples"
            )
            
            # Normalize rewards when there is meaningful variance.
            # FIX: Only normalize if there is variance. If all rewards are -5.0 (failure),
            # subtracting mean makes them all 0.0, killing the negative signal.
            if len(rewards) > 1 and rewards.std() > 1e-4:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                # If rewards are constant (e.g. all -5.0), do NOT center them.
                # Keep them as is so the agent knows it's failing.
                pass

            # Concatenate buffered tensors and move them to the target device.
            curr_map = torch.cat(self.buffer["curr_map"]).to(device)
            mask = torch.cat(self.buffer["mask"]).to(device)
            action = torch.cat(self.buffer["action"]).to(device)
            stats_action = torch.cat(self.buffer["stats_action"]).to(device)
            old_logprob = torch.cat(self.buffer["logprob"]).to(device)
            old_value = torch.cat(self.buffer["value"]).to(device).squeeze()
            topk_mask = torch.cat(self.buffer["topk_mask"]).to(device)
            stats_topk_mask = torch.cat(self.buffer["stats_topk_mask"]).to(device)

            # Gather HistoryEncoder inputs.
            prev_map = torch.cat(self.buffer["prev_map"]).to(device)
            prev_heat_terrain = torch.cat(self.buffer["prev_heat"]).to(device)
            prev_heat_stats = torch.cat(self.buffer["stats_heat"]).to(device)     # [B, 16]

            # --- Step 2: Advantage normalization ---
            # Advantage = reward - value estimate.
            advantages = rewards - old_value.detach()
            
            # FIX: Only normalize advantages if they have variance.
            if len(advantages) > 1 and advantages.std() > 1e-4:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            last_loss = 0.0

            # --- Step 3: PPO K-epoch training loop ---
            for i in range(self.K_epochs):
                if self.encoder is None:
                    ctx = None
                    eval_stats_heat = prev_heat_stats
                else:
                    global_ctx = self._compute_global_context_dual(prev_map, prev_heat_terrain, prev_heat_stats)
                    if global_ctx.size(0) == curr_map.size(0):
                        ctx = global_ctx
                    else:
                        ctx = global_ctx.repeat(curr_map.size(0), 1)
                    eval_stats_heat = prev_heat_stats

                # action_tuple: (terrain_action, stats_action)
                logp_terrain, logp_stats, value, entropy = self.policy.evaluate(
                    curr_map,
                    ctx,
                    (action, stats_action),
                    mask,
                    target_topk_mask=topk_mask,
                    target_stats_topk_mask=stats_topk_mask,
                    stats_heat=eval_stats_heat,
                )

                # Joint LogProb
                total_logp = logp_terrain.sum(dim=(1, 2)) + logp_stats
                value = value.squeeze()

                ratio = torch.exp(total_logp - old_logprob.detach())

                # Compute the clipped PPO surrogate loss.
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # 4. Loss composition:
                # - policy loss increases the probability of rewarding actions
                # - value loss improves critic estimates
                # - entropy loss preserves exploration
                loss_policy = -torch.min(surr1, surr2).mean()
                loss_value = 0.5 * self.mse(value, rewards)
                loss_entropy = -current_entropy_coef * entropy.mean()

                total_loss = loss_policy + loss_value + loss_entropy
                
                # --- Numerical safety check ---
                if torch.isnan(total_loss):
                    print(f"[GeneratorPPO] Warning: NaN detected in Epoch {i}. Stopping update.")
                    return 0.0, 0.0 

                # --- Step 4: Backpropagation and gradient clipping ---
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients for both the policy and the HistoryEncoder.
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                if self.encoder is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)

                
                self.optimizer.step()

                last_loss = total_loss.item()

            # --- Step 5: State synchronization and cleanup ---
            # Synchronize the old policy. The rollout window was trimmed before
            # optimization so every sample used here belongs to that window.
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            # Return entropy for monitoring.
            return last_loss, entropy.mean().item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        save_dict = {"policy": self.policy.state_dict()}
        if self.encoder is not None:
            save_dict["encoder"] = self.encoder.state_dict()
        torch.save(save_dict, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy_old.load_state_dict(ckpt["policy"])
        if self.encoder is not None and "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"])
