import torch
import torch.nn as nn
import torch.optim as optim

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
        num_actions=11,
        # ratio=0.25, # removed
        top_k_features=16,
        ablation_type="none",

    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.context_dim = context_dim
        # self.ratio = ratio # removed
        self.top_k_features = top_k_features

        # history encoder
        

        # === Networks ===
        if ablation_type == "no_history":
            self.encoder = None  # No history encoder needed for this ablation
        else:
            self.encoder = HistoryEncoder(context_dim=context_dim,    
            emb_dim=his_emb_dim).to(device)

        self.policy = MapEditorActorCritic(
            context_dim=context_dim,
            num_actions=num_actions,
            ablation_type=ablation_type,
        ).to(device)

        self.policy_old = MapEditorActorCritic(
            context_dim=context_dim,
            num_actions=num_actions,
            ablation_type=ablation_type,
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
            "action": [],
            "logprob": [],
            "value": [],
            "reward": [],
            "topk_mask": [],
        }
        self.last_mean_reward = 0.0

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def _compute_global_context(self, prev_map, prev_heat, top_k_features=16):
        """
        计算全局上下文向量 (Global Context Aggregation):
        从本轮所有样本中提炼出“失败模式的并集”，并过滤噪声，为生成器提供精准的训练目标。

        参数:
            prev_map: 上一轮的地图布局 [B, 3, H, W]
            prev_heat: 上一轮的预测误差图 (Error Heatmap) [B, 1, H, W]
            top_k_features: 显著性过滤阈值，只保留最强的 K 个失败信号 (推荐 16)
        
        返回:
            v_ctx: 形状为 [1, context_dim] 的稳定上下文向量
        """
        
        # === Step 1: 独立特征提取 ===
        # 利用 HistoryEncoder 提取每个样本的原始失败特征向量 [B, context_dim]
        # 注意：HistoryEncoder 输出层需为 ReLU，确保特征全为非负数
        ctx = self.encoder(prev_map, prev_heat) 
        B = ctx.size(0)

        # === Step 2: 局部误差得分 (Score Calculation) ===
        # 我们使用误差图的“最大值”而不是“均值”作为评分。
        # 语义：只要地图中有一个点让智能体彻底由于逻辑错误而崩溃，
        # 哪怕地图其他地方很完美，这个样本也具有极高的“失败模式”提取价值。
        with torch.no_grad():
            # 将 [B, 1, H, W] 展平并取每个样本的最大误差值
            # scores 形状: [B, 1]
            scores = prev_heat.view(B, -1).max(dim=1)[0].view(B, 1)

            # 归一化得分，防止数值过大影响梯度，同时增强 Batch 内的对比度
            scores = scores / (scores.max() + 1e-6)

        # === Step 3: 误差门控 (Error Gating) ===
        # 只有预测误差大的样本，其特征向量才会被放大。
        # 如果样本预测很准（score接近0），其特征会被压制，不进入全局上下文。
        gated_ctx = ctx * scores # [B, context_dim]

        # === Step 4: 提取失败模式并集 (Union via Max-Pooling) ===
        # 跨 Batch 维度取最大值。
        # 结果中的每一维都代表了在本轮所有失败关卡中，该特定失败模式出现的“最大强度”。
        # v_ctx 形状: [1, context_dim]
        v_ctx, _ = torch.max(gated_ctx, dim=0, keepdim=True)

        # === Step 5: 显著性过滤 (Top-K Sparsification) ===
        # 目的：防止“满屏红灯”，让生成器每一轮只专注解决最核心的几个弱点。
        # 这能显著提升 PPO 算法的训练稳定性，建立清晰的因果关联。
        if v_ctx.size(1) > top_k_features:
            # 找到第 K 个最强信号的大小
            top_val, _ = torch.topk(v_ctx, k=top_k_features, dim=1)
            min_val = top_val[:, -1:] # 第 K 个值作为门槛
            
            # 硬过滤：低于门槛的信号直接置 0
            mask = (v_ctx >= min_val).float()
            v_ctx = v_ctx * mask

        # === Step 6: 数值归一化 (Normalization) ===
        # 语义：无论这轮失败有多惨烈，传给生成器的信号量级应当是稳定的。
        # 这能防止 Generator 的权重由于上下文数值的剧烈波动而跳变。
        v_ctx = F.normalize(v_ctx, p=2, dim=1) 

        return v_ctx

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, base_map, prev_data, mask, max_edits):
        """
        Samples generator edit actions conditioned on the current map and
        historical rollout context.
        inputs:
            base_map: [B, C, H, W]
            immutable_mask:     [B, 1, H, W]
            prev_data: (prev_map, prev_heat) or None
        """

        B = base_map.size(0)

        if self.encoder is None:   # no_history ablation
            ctx = None
            global_ctx = None
        else:
            if prev_data is None:
                global_ctx = torch.zeros(1, self.context_dim, device=device)
            else:
                global_ctx = self._compute_global_context(
                    *prev_data,
                    self.top_k_features
                )
            ctx = global_ctx.repeat(B, 1)

        action, logprob_map, value, topk_mask = self.policy_old.act(
            base_map, ctx, mask, max_edits
        )

        logprob = logprob_map.sum(dim=(1, 2))  # [B]

        return action, logprob, value, topk_mask, global_ctx

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------
    def save_buffer(
        self,
        curr_map,
        prev_data,
        mask,
        action,
        logprob,
        value,
        reward,
        topk_mask,
    ):
        """
        All tensors are [1, ...]
        """

        self.buffer["curr_map"].append(curr_map.cpu())
        self.buffer["mask"].append(mask.cpu())
        self.buffer["action"].append(action.cpu())
        self.buffer["logprob"].append(logprob.cpu())
        self.buffer["value"].append(value.cpu())
        self.buffer["reward"].append(float(reward))  # ★ FIX: 强制标量
        self.buffer["topk_mask"].append(topk_mask.cpu())

        if prev_data is None:
            B, _, H, W = curr_map.shape
            self.buffer["prev_map"].append(torch.zeros((B, 3, H, W)))
            self.buffer["prev_heat"].append(torch.zeros((B, 1, H, W)))

        else:
            pm, ph = prev_data
            self.buffer["prev_map"].append(pm.cpu())
            self.buffer["prev_heat"].append(ph.cpu())

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k].clear()

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self):
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

            # --- Step 1: 基础数据准备与归一化 ---
            rewards = torch.tensor(self.buffer["reward"], device=device)

            # [Monitor] 打印平均奖励，用于判断策略是否收敛 (Mean Reward should increase)
            mean_reward = rewards.mean().item()
            self.last_mean_reward = mean_reward
            print(f"[GeneratorPPO] Mean Reward: {mean_reward:.4f}")
            
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
            old_logprob = torch.cat(self.buffer["logprob"]).to(device)
            old_value = torch.cat(self.buffer["value"]).to(device).squeeze()
            topk_mask = torch.cat(self.buffer["topk_mask"]).to(device)

            # 获取用于 HistoryEncoder 的素材
            prev_map = torch.cat(self.buffer["prev_map"]).to(device)
            prev_heat = torch.cat(self.buffer["prev_heat"]).to(device)

            # --- Step 2: 优势函数归一化 (Advantage Normalization) ---
            # 优势 = 实际奖励 - 预测价值。这是 PPO 稳定性的基石。
            advantages = rewards - old_value.detach()
            
            # FIX: Only normalize advantages if they have variance.
            if len(advantages) > 1 and advantages.std() > 1e-4:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            last_loss = 0.0

            # --- Step 3: PPO K-Epochs 训练循环 ---
            for i in range(self.K_epochs):
                
                # 核心改动：调用统一的聚合函数，建立从奖励到 Encoder 参数的梯度链路
                # 这样更新过程不仅优化了 Policy (MapEditor)，也同时进化了 HistoryEncoder
                if self.encoder is None:
                    ctx = None
                else:
                    global_ctx = self._compute_global_context(
                        prev_map,
                        prev_heat,
                        top_k_features=16
                    )
                    ctx = global_ctx.repeat(curr_map.size(0), 1)


                # 评估当前最新策略下的动作概率和价值
                logprob_map, value, entropy = self.policy.evaluate(
                    curr_map, ctx, action, mask, target_topk_mask=topk_mask
                )

                # PPO 概率计算 (空间维度求和，得到整个地图编辑动作的总 log_prob)
                logprob = logprob_map.sum(dim=(1, 2))
                value = value.squeeze()

                # 计算概率比率 (Ratio)
                ratio = torch.exp(logprob - old_logprob.detach())

                # 计算 PPO 裁剪后的 Surrogate Loss (防止策略更新过猛)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # 4. 损失函数组合
                # - Policy Loss: 让奖励高的动作概率变大
                # - Value Loss: 让 Critic 估分更准 (MSE)
                # - Entropy Loss: 鼓励探索，防止生成器退化成只会出一种题 (Increased to 0.8 to break collapse)
                loss_policy = -torch.min(surr1, surr2).mean()
                loss_value = 0.5 * self.mse(value, rewards)
                loss_entropy = -0.8 * entropy.mean()

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
            self.clear_buffer()
            
            # [MODIFIED] Return entropy for monitoring
            return last_loss, entropy.mean().item()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "encoder": self.encoder.state_dict(),
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy_old.load_state_dict(ckpt["policy"])
        self.encoder.load_state_dict(ckpt["encoder"])
