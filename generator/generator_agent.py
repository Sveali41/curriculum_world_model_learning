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
        self.env_type = env_type
        # self.ratio = ratio # removed
        self.top_k_features = top_k_features
        self.is_bipedal = (env_type == "bipedalwalker")
        self.policy_context_dim = context_dim if self.is_bipedal else context_dim * 2

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
        ).to(device)

        self.policy_old = MapEditorActorCritic(
            num_actions=num_actions,
            context_dim=self.policy_context_dim,
            ablation_type=ablation_type,
            env_type=env_type,
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

        self.optimizer = optim.Adam(optim_params)

        self.mse = nn.MSELoss()

        # === PPO Buffer (统一单样本维度) ===
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
    def _compute_global_context_dual(self, prev_map, terrain_heat, stats_heat, top_k_features=16):
        """
        聚合物理布局失败特征 (Spatial) 和 物资数值失败特征 (Inventory).
        For Crafter/MiniGrid we keep both:
        - local per-sample history context
        - global batch summary context
        """
        # 使用更新后的 HistoryEncoder 提取每个样本的综合失败特征
        ctx = self.encoder(prev_map, terrain_heat, stats_heat) # [B, context_dim]

        if self.is_bipedal:
            return F.normalize(ctx, p=2, dim=1)
        
        local_ctx = F.normalize(ctx, p=2, dim=1)

        # 跨 Batch 取并集 (Max-Pooling)
        v_ctx, _ = torch.max(ctx, dim=0, keepdim=True) # [1, context_dim]

        # 显著性过滤 (Top-K Sparsification)
        if v_ctx.size(1) > top_k_features:
            top_val, _ = torch.topk(v_ctx, k=top_k_features, dim=1)
            min_val = top_val[:, -1:]
            mask = (v_ctx >= min_val).float()
            v_ctx = v_ctx * mask

        # 数值归一化
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
                # 初始状态：空地图、空热图、空背包热图
                H, W = base_map.size(2), base_map.size(3)
                # pm: prev_map, pht: prev_heat_terrain, phs: prev_heat_stats
                pm = torch.zeros((1, 3, H, W), device=device)
                pht = torch.zeros((1, 1, H, W), device=device)
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
            self.buffer["prev_heat"].append(torch.zeros((B, 1, H, W)))
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
            PPO 更新逻辑：
            1. 从 Buffer 中提取并处理本轮收集到的所有经验。
            2. 重新计算上下文 (Global Context) 以便更新 HistoryEncoder 的参数。
            3. 计算 PPO 裁剪损失、价值损失以及熵损失。
            4. 执行反向传播并同步策略。
            """
            # --- Step 0: 安全检查 ---
            if len(self.buffer["curr_map"]) == 0:
                print("[GeneratorPPO] Warning: Buffer is empty, skipping update.")
                return 0.0, 0.0
            if self.current_round_count > 0:
                self.round_lengths.append(self.current_round_count)
                self.current_round_count = 0

            # --- Step 1: 基础数据准备与归一化 ---
            rewards = torch.tensor(self.buffer["reward"], device=device)

            # [Monitor] 打印平均奖励，用于判断策略是否收敛 (Mean Reward should increase)
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
            
            # 奖励归一化：保证奖励量级稳定
            # FIX: Only normalize if there is variance. If all rewards are -5.0 (failure),
            # subtracting mean makes them all 0.0, killing the negative signal.
            if len(rewards) > 1 and rewards.std() > 1e-4:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                # If rewards are constant (e.g. all -5.0), do NOT center them.
                # Keep them as is so the agent knows it's failing.
                pass

            # 拼接 Buffer 里的张量并搬运到 GPU
            curr_map = torch.cat(self.buffer["curr_map"]).to(device)
            mask = torch.cat(self.buffer["mask"]).to(device)
            action = torch.cat(self.buffer["action"]).to(device)
            stats_action = torch.cat(self.buffer["stats_action"]).to(device)
            old_logprob = torch.cat(self.buffer["logprob"]).to(device)
            old_value = torch.cat(self.buffer["value"]).to(device).squeeze()
            topk_mask = torch.cat(self.buffer["topk_mask"]).to(device)
            stats_topk_mask = torch.cat(self.buffer["stats_topk_mask"]).to(device)

            # 获取用于 HistoryEncoder 的素材
            prev_map = torch.cat(self.buffer["prev_map"]).to(device)
            prev_heat_terrain = torch.cat(self.buffer["prev_heat"]).to(device) # [B, 1, H, W]
            prev_heat_stats = torch.cat(self.buffer["stats_heat"]).to(device)     # [B, 16]

            # --- Step 2: 优势函数归一化 (Advantage Normalization) ---
            # 优势 = 实际奖励 - 预测价值。这是 PPO 稳定性的基石。
            advantages = rewards - old_value.detach()
            
            # FIX: Only normalize advantages if they have variance.
            if len(advantages) > 1 and advantages.std() > 1e-4:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            last_loss = 0.0

            # --- Step 3: PPO K-Epochs 训练循环 ---
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

                # 计算 PPO 裁剪后的 Surrogate Loss (防止策略更新过猛)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # 4. 损失函数组合
                # - Policy Loss: 让奖励高的动作概率变大
                # - Value Loss: 让 Critic 估分更准 (MSE)
                # - Entropy Loss: 不要太大，否则会强迫生成器永远随机乱下棋 (Reduced from 0.05 to 0.01 for stability)
                loss_policy = -torch.min(surr1, surr2).mean()
                loss_value = 0.5 * self.mse(value, rewards)
                loss_entropy = -current_entropy_coef * entropy.mean()

                total_loss = loss_policy + loss_value + loss_entropy
                
                # --- 数值安全性检查 ---
                if torch.isnan(total_loss):
                    print(f"[GeneratorPPO] Warning: NaN detected in Epoch {i}. Stopping update.")
                    return 0.0, 0.0 

                # --- Step 4: 反向传播与梯度裁剪 ---
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 同时裁剪 Policy 网络和 HistoryEncoder 的梯度，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

                if self.encoder is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)

                
                self.optimizer.step()

                last_loss = total_loss.item()

            # --- Step 5: 状态同步与清理 ---
            # 更新完成后，将旧策略同步到最新状态，并清空 Buffer 迎接下一轮采样
            self.policy_old.load_state_dict(self.policy.state_dict())
            self._trim_buffer_to_recent_rounds()
            
            # [MODIFIED] Return entropy for monitoring
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
