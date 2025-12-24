import torch
import torch.nn as nn
import torch.optim as optim

from generator.generator_network import MapEditorActorCritic
from generator.history_encoder import HistoryEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneratorPPO:
    """
    PPO-based Generator with a learnable History Encoder.
    """

    def __init__(
        self,
        his_emb_dim=16,
        context_dim=64,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        num_actions=11,
        ratio=0.25,

    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.context_dim = context_dim
        self.ratio = ratio

        # history encoder
        

        # === Networks ===
        self.encoder = HistoryEncoder(context_dim=context_dim,    
            emb_dim=his_emb_dim).to(device)

        self.policy = MapEditorActorCritic(
            context_dim=context_dim,
            num_actions=num_actions,
        ).to(device)

        self.policy_old = MapEditorActorCritic(
            context_dim=context_dim,
            num_actions=num_actions,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # === Optimizer (encoder + policy) ===
        self.optimizer = optim.Adam(
            [
                {"params": self.encoder.parameters(), "lr": lr_actor},
                {"params": self.policy.stem.parameters(), "lr": lr_actor},
                {"params": self.policy.res_blocks.parameters(), "lr": lr_actor},
                {"params": self.policy.emb_obj.parameters(), "lr": lr_actor},
                {"params": self.policy.emb_color.parameters(), "lr": lr_actor},
                {"params": self.policy.emb_state.parameters(), "lr": lr_actor},
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.mse = nn.MSELoss()

        # === PPO Buffer (统一单样本维度) ===
        self.buffer = {
            "curr_map": [],
            "prev_map": [],
            "prev_heat": [],
            "prev_attn": [],
            "mask": [],
            "action": [],
            "logprob": [],
            "value": [],
            "reward": [],
        }

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    def _compute_global_context(self, prev_map, prev_heat, prev_attn, ratio=0.25):
        """
        Aggregate batch-level history using top-k pooling to capture
        the union of dominant failure patterns while suppressing noise.

        prev_*: [B, C, H, W]
        return: [1, context_dim]
        """
        ctx = self.encoder(prev_map, prev_heat, prev_attn)  # [B, D]
        B = ctx.size(0)
        k = int(ratio * B)
        topk_vals, _ = torch.topk(ctx, k=min(k, ctx.size(0)), dim=0)
        return topk_vals.mean(dim=0, keepdim=True)

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
            prev_data: (prev_map, prev_heat, prev_attn) or None
        """

        B = base_map.size(0)

        if prev_data is None:
            global_ctx = torch.zeros(1, self.context_dim, device=device)
        else:
            global_ctx = self._compute_global_context(*prev_data, self.ratio)  # [1, D]

        ctx = global_ctx.repeat(B, 1) # the context for each sample in the batch

        action, logprob_map, value = self.policy_old.act(
            base_map, ctx, mask, max_edits
        )

        logprob = logprob_map.sum(dim=(1, 2))  # [B]

        return action, logprob, value, global_ctx

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

        if prev_data is None:
            B, _, H, W = curr_map.shape
            self.buffer["prev_map"].append(torch.zeros((B, 3, H, W)))
            self.buffer["prev_heat"].append(torch.zeros((B, 1, H, W)))
            self.buffer["prev_attn"].append(torch.zeros((B, 1, H, W)))
        else:
            pm, ph, pa = prev_data
            self.buffer["prev_map"].append(pm.cpu())
            self.buffer["prev_heat"].append(ph.cpu())
            self.buffer["prev_attn"].append(pa.cpu())

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k].clear()

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------
    def update(self):
        rewards = torch.tensor(self.buffer["reward"], device=device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        curr_map = torch.cat(self.buffer["curr_map"]).to(device)
        mask = torch.cat(self.buffer["mask"]).to(device)
        action = torch.cat(self.buffer["action"]).to(device)
        old_logprob = torch.cat(self.buffer["logprob"]).to(device)
        old_value = torch.cat(self.buffer["value"]).to(device).squeeze()

        prev_map = torch.cat(self.buffer["prev_map"]).to(device)
        prev_heat = torch.cat(self.buffer["prev_heat"]).to(device)
        prev_attn = torch.cat(self.buffer["prev_attn"]).to(device)

        advantages = rewards - old_value.detach()

        last_loss = 0.0

        for _ in range(self.K_epochs):
            ctx_batch = self.encoder(prev_map, prev_heat, prev_attn)
            global_ctx = ctx_batch.mean(dim=0, keepdim=True)
            ctx = global_ctx.repeat(curr_map.size(0), 1)

            logprob_map, value, entropy = self.policy.evaluate(
                curr_map, ctx, action, mask
            )

            logprob = logprob_map.sum(dim=(1, 2))
            value = value.squeeze()

            ratio = torch.exp(logprob - old_logprob.detach())

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse(value, rewards)
                - 0.01 * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            last_loss = loss.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

        return last_loss

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
