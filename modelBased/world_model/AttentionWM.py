import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Union
from modelBased.common import utils
from domain.minigrid import minigrid_support as minigrid_utils
from domain.crafter.crafter_support import (
    crafter_classification_loss,
    crafter_reconstruct_from_logits,
    visualize_crafter_wm,
)
from . import AttentionWM_support
from . import Embedding_support
from . import MLP_support
import pandas as pd


class AttentionWorldModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.mask_size = hparams.attention_mask_size
        self.channel, self.row, self.col = hparams.grid_shape
        self.lr= hparams.lr
        self.weight_decay = hparams.wd
        self.visualizationFlag = hparams.visualization
        self.visualize_every = hparams.visualize_every
        self.step_counter = 0  
        self.data_type = hparams.data_type
        self.ewc_ratio = getattr(hparams, "ewc_ratio", 0.2)           # 目标占比：EWC ≈ 20% * obs_loss；想手动控制就在 yaml 里设为 null
        self.lambda_ema = getattr(hparams, "lambda_ema", 0.1)         # λ 的 EMA 平滑系数
        self.lambda_ewc_min = getattr(hparams, "lambda_ewc_min", 1e-4)
        self.lambda_ewc_max = getattr(hparams, "lambda_ewc_max", 1e3)
        self.lambda_ewc = float(getattr(hparams, "lambda_ewc", 1.0))
        self.loss_accumulator = [[[] for _ in range(self.col)] for _ in range(self.row)]
        self.loss_map_result = None



        # 慢速外环（漂移）相关
        self.warmup_steps = getattr(hparams, "warmup_steps", 100)
        self.drift_cooldown = getattr(hparams, "drift_cooldown", 200)  # 多久允许调整一次
        self._last_drift_update_step = -10**9
        self.drift_threshold = getattr(hparams, "drift_threshold", 1e-3)
        self.fisher = 0
        self.old_params = None
        self.env_type = hparams.env_type
        self.frame_stack = getattr(hparams, "frame_stack", 1)
        MODEL_MAPPING = {
            'attention': AttentionWM_support.AttentionModule,
            'embedding': Embedding_support.EmbeddingModule,
            'mlp': MLP_support.SimpleNNModule
        }
        # 初始化模型
        module_class = MODEL_MAPPING.get(hparams.model_type.lower())
        if module_class is not None:
            self.model = module_class(
                hparams.data_type, 
                hparams.grid_shape, 
                hparams.attention_mask_size, 
                hparams.embed_dim, 
                hparams.num_heads,
                env_type=self.env_type,
                frame_stack=self.frame_stack
            )
        else:
            print(f"Model type: {hparams.model_type} not supported")
            exit()
        

        if hparams.freeze_weight:
            utils.load_model_weight(self.model, hparams.model_save_path)
        self.loss = nn.MSELoss() # nn.SmoothL1Loss()
        self.visual_func = minigrid_utils.Visualization(hparams)
        self.save_hyperparameters(hparams)


    def save_old_params(self):
        """Save current model parameters for EWC, moved to model's device."""
        device = next(self.parameters()).device
        old_params = {k: v.clone().detach().cpu() 
              for k, v in self.state_dict().items()}
        return old_params
    
    # def load_old_params(self, old_params):
    #     """加载旧参数并应用到当前模型"""
    #     for n, p in self.named_parameters():
    #         if n in old_params:
    #             p.data.copy_(old_params[n])  # 使用旧参数的值替换当前参数

    def compute_avg_param_drift(self) -> float:
        drift_sum = 0.0
        num = 0
        for name, param in self.named_parameters():
            if self.old_params is not None and name in self.old_params:
                drift = (param - self.old_params[name].to(param.device)).pow(2).mean()
                drift_sum += drift.item()
                num += 1
        avg_drift = drift_sum / max(num, 1)
        return avg_drift

    def update_lambda_ewc_by_ratio(self, obs_loss: torch.Tensor, ewc_raw: torch.Tensor, r: float | None):
        """
        [DEPRECATED / FIXED]
        Originally attempted to dynamically scale lambda based on loss ratio.
        Removed to ensure stability. Now uses fixed self.lambda_ewc.
        """
        pass 
        # Stability fix: Do NOT dynamically change lambda based on inverse of drift.
        # This caused exploding updates or vanishing constraints.
        # We rely on fixed lambda_ewc set in config.

    def update_lambda_ewc(self, avg_drift: float):
        """
        慢速外环：基于参数漂移 avg_drift 的 λ 调整（与快速占比控制器互补）
        - warmup 自动学习 drift_threshold（当 hparams.drift_threshold 为 None 时）
        - 冷却时间 cooldown：减少与快速控制器的相互干扰
        - 滞回区间 hysteresis：超出阈值范围才调整
        - 对数域微调：上下调更平滑，再做 EMA 平滑与边界裁剪
        """
        import math
        import torch

        # —— 保护性默认值（若没设定）——
        if not hasattr(self, "lambda_ewc"):
            self.lambda_ewc = float(getattr(self.hparams, "lambda_ewc", 1.0))
        if not hasattr(self, "_drift_values"):
            self._drift_values = []
        if not hasattr(self, "_last_drift_update_step"):
            self._last_drift_update_step = -10**9

        # —— 读超参（允许在 hparams 或实例属性上覆盖）——
        warmup_steps = getattr(self.hparams, "warmup_steps", getattr(self, "warmup_steps", 100))
        cooldown = getattr(self.hparams, "drift_cooldown", getattr(self, "drift_cooldown", 200))
        hi_ratio = getattr(self.hparams, "drift_hi", 1.10)  # 高于阈值 110% 才上调
        lo_ratio = getattr(self.hparams, "drift_lo", 0.70)  # 低于阈值 70% 才下调
        up_step = getattr(self.hparams, "drift_up_step", 0.10)  # 对数域上调步长
        down_step = getattr(self.hparams, "drift_down_step", 0.05)  # 对数域下调步长
        lam_min = getattr(self.hparams, "lambda_ewc_min", getattr(self, "lambda_ewc_min", 1e-4))
        lam_max = getattr(self.hparams, "lambda_ewc_max", getattr(self, "lambda_ewc_max", 1e3))
        lam_ema = getattr(self.hparams, "lambda_ema", getattr(self, "lambda_ema", 0.1))

        # —— 记录漂移日志 —— 
        self.log("train/avg_param_drift", avg_drift)

        # —— warmup：若阈值为 None，则用前 warmup_steps 个漂移的均值做阈值 —— 
        if getattr(self, "drift_threshold", None) is None:
            self._drift_values.append(float(avg_drift))
            if len(self._drift_values) >= warmup_steps:
                self.drift_threshold = float(sum(self._drift_values) / len(self._drift_values))
                print(f"[Auto-tuned] drift_threshold set to {self.drift_threshold:.6f}")
            # warmup 期间不调 λ
            return

        # —— 冷却：避免频繁与快速控制器打架 —— 
        if self.global_step - self._last_drift_update_step < cooldown:
            return

        # —— 滞回区间：超过才调 —— 
        hi = self.drift_threshold * hi_ratio
        lo = self.drift_threshold * lo_ratio

        lam = float(max(self.lambda_ewc, lam_min))
        lam_log = math.log(lam)
        changed = False

        if avg_drift > hi:
            lam_log += up_step
            changed = True
        elif avg_drift < lo:
            lam_log -= down_step
            changed = True

        if changed:
            lam_new = math.exp(lam_log)
            # 边界裁剪
            lam_new = float(torch.clamp(torch.tensor(lam_new), lam_min, lam_max))
            # EMA 平滑
            self.lambda_ewc = (1.0 - lam_ema) * self.lambda_ewc + lam_ema * lam_new
            # 更新时间戳
            self._last_drift_update_step = self.global_step

    def load_old_params(self, old_params):
        device = next(self.parameters()).device
        self.old_params = {k: v.to(device) for k, v in old_params.items()}

    # def compute_fisher(self, dataloader, samples, scale_factor):
    #     fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
    #     self.eval()
    #     device = next(self.parameters()).device
    #     count = 0
    #     for i, batch in enumerate(dataloader):
    #         if count >= samples:
    #             break
    #         self.zero_grad()

    #         obs, act, obs_next, info, obs_masked = self.preprocess_batch(batch)
    #         obs = obs.to(device).float()
    #         act = act.to(device)
    #         obs_next = obs_next.to(device).float()
    #         obs_pred, _ = self(obs, act, info)
    #         loss = scale_factor * self.loss_function_weight(obs_pred, obs_next, obs_masked)['loss_obs']
    #         loss.backward(retain_graph=False)
    #         for n, p in self.named_parameters():
    #             if p.grad is not None:
    #                 fisher[n] += p.grad.detach().pow(2)
    #         count += 1
    #     for n in fisher:
    #         fisher[n] /= count
    #     # for k in fisher:
    #     #     fisher[k] = torch.sqrt(fisher[k] + 1e-8)
    #     #     fisher[k] *= 5
    #     # Fisher normalization by mean
    #     for k in fisher:
    #         mean_val = fisher[k].mean()
    #         fisher[k] = scale_factor * fisher[k] / (mean_val + 1e-8)

    #     all_f = torch.cat([f.flatten() for f in fisher.values()])
    #     print(f"[Fisher] mean={all_f.mean():.3e}, max={all_f.max():.3e}, min={all_f.min():.3e}")
    #     return fisher

    def compute_fisher(self, dataloader, samples, scale_factor):
        """
        Correct diagonal Fisher Information Matrix (FIM) estimation.
        Formula: F = E[ (\nabla \log p(x))^2 ]
        
        Implementation:
        1. Iterate through the dataloader.
        2. For each batch, iterate through INDIVIDUAL samples.
        3. Compute gradient for single sample -> square it -> accumulate.
        4. Average over total samples.
        """
        import torch
        self.eval() 
        device = next(self.parameters()).device

        fisher = {n: torch.zeros_like(p, dtype=torch.float32, device=device)
                for n, p in self.named_parameters() if p.requires_grad}

        count = 0
        total_samples_target = int(samples)
        
        print(f"[Fisher] Starting computation for ~{total_samples_target} samples...")

        for i, batch in enumerate(dataloader):
            if count >= total_samples_target:
                break
            
            # 1. Preprocess batch
            obs, act, obs_next, info, obs_masked, _, inv, inv_next = self.preprocess_batch(batch)
            obs = obs.to(device, dtype=torch.float32)
            act = act.to(device)
            obs_next = obs_next.to(device, dtype=torch.float32)
            if obs_masked is not None:
                obs_masked = obs_masked.to(device)
            if inv is not None:
                inv = inv.to(device)
            if inv_next is not None:
                inv_next = inv_next.to(device)
            
            # 2. Iterate over samples in the batch
            batch_size = obs.shape[0]
            for b in range(batch_size):
                if count >= total_samples_target:
                    break
                    
                self.zero_grad(set_to_none=True)

                # Extract single sample (keep dim 0 for model compatibility)
                s_obs = obs[b:b+1]
                s_act = act[b:b+1]
                s_info = {k: v[b:b+1] for k, v in info.items()} if info is not None else None
                s_obs_next = obs_next[b:b+1]
                s_obs_masked = obs_masked[b:b+1] if obs_masked is not None else None
                s_inv = inv[b:b+1] if inv is not None else None
                s_inv_next = inv_next[b:b+1] if inv_next is not None else None

                # Forward & Backward (fp32)
                with torch.cuda.amp.autocast(enabled=False):
                    pred, _, s_inv_pred = self(s_obs, s_act, s_info, inv=s_inv)
                    loss_dict = self.loss_function_weight(pred, s_obs_next, s_obs_masked, obs_prev=s_obs)
                    loss_sample = loss_dict['loss_obs']
                    
                    if self.env_type == 'crafter' and s_inv_pred is not None and s_inv_next is not None:
                        if s_inv is not None:
                            inv_diff = torch.abs(s_inv_next - s_inv).float()
                            # If an inventory element changes, hit it with x100 magnitude
                            inv_w = 1.0 + (inv_diff > 1e-5).float() * 100.0
                            loss_inv = (F.mse_loss(s_inv_pred, s_inv_next.float(), reduction='none') * inv_w).mean()
                        else:
                            loss_inv = F.mse_loss(s_inv_pred, s_inv_next.float())
                        loss_sample = loss_sample + 10.0 * loss_inv
                
                loss_sample.backward()

                # Accumulate squares
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        # Square the gradient of this SINGLE sample
                        g2 = p.grad.detach().float().pow(2)
                        fisher[n] += g2
                
                count += 1
        
        if count == 0:
             print("[Fisher] Warning: No samples processed!")
             return fisher

        # 3. Normalize by N (Average)
        for n in fisher:
            fisher[n] /= float(count)

        # 4. Standardize Fisher (Normalize to Mean=1.0) to make lambda_ewc scale-invariant
        with torch.no_grad():
            all_vals = torch.cat([f.flatten() for f in fisher.values()])
            mean_val = all_vals.mean()
            max_val = all_vals.max()
            
            print(f"[Fisher] Computed with {count} samples. Raw Mean={mean_val:.3e}, Max={max_val:.3e}")
            
            if mean_val > 1e-20:
                scale = 1.0 / mean_val
                # Apply normalization FIRST
                for n in fisher:
                    fisher[n] *= scale
                print(f"[Fisher] Normalized to Mean=1.0. Applied scale: {scale:.3e}")
                
                # THEN apply user scale_factor if needed (usually 1.0 now)
                if scale_factor != 1.0:
                    for n in fisher:
                        fisher[n] *= scale_factor
                    print(f"[Fisher] Applied extra config scale_factor: {scale_factor}")
            else:
                print("[Fisher] Warning: Fisher values are essentially zero. Check gradients!")

        # Move to CPU for storage
        fisher = {k: v.detach().cpu() for k, v in fisher.items()}
        return fisher


    

    # def ewc_loss(self, lambda_ewc):
    #     if self.fisher is None or self.old_params is None:
    #         return torch.tensor(0.0, device=next(self.parameters()).device)
        
    #     device = next(self.parameters()).device
    #     loss = torch.tensor(0.0, device=device)  
    #     for n, p in self.named_parameters():
    #         if n in self.fisher and n in self.old_params:
    #             fisher = self.fisher[n].to(device)
    #             p_old = self.old_params[n].to(device)
    #             loss += (fisher * (p - p_old).pow(2)).sum()

    #     return lambda_ewc * loss

    def set_consolidation(self, old_params: dict, fisher: dict, load_weights: bool = True):
        """
        设置 EWC 的“锚点”信息（旧参数 + Fisher），并可选地加载旧参数权重到当前模型。

        Args:
            old_params (dict): 上一阶段保存的模型参数 (state_dict)
            fisher (dict): Fisher 信息矩阵
            load_weights (bool): 是否将旧参数直接加载到当前模型
        """
        # ----------------------------------------------------------
        # (1) 旧参数部分
        # ----------------------------------------------------------
        if old_params is not None:
            # 保存旧参数为 CPU 版本 (float32)
            self.old_params = {k: v.detach().cpu().float() for k, v in old_params.items()}

            if load_weights:
                # 尝试将旧参数加载进当前模型
                current_state = self.state_dict()
                updated_state = {}

                loaded_keys, skipped_keys = [], []
                for k, v in old_params.items():
                    if k in current_state and current_state[k].shape == v.shape:
                        updated_state[k] = v.clone().detach()
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)

                # 执行加载（严格性关闭以防形状不匹配）
                current_state.update(updated_state)
                self.load_state_dict(current_state, strict=False)

                print(f"[EWC] Loaded {len(loaded_keys)} parameters from previous task "
                    f"(skipped {len(skipped_keys)} mismatched keys).")
            else:
                print("[EWC] old_params received but model weights not loaded (load_weights=False).")

        else:
            self.old_params = None
            print("[EWC] No old_params provided — starting from scratch.")

        # ----------------------------------------------------------
        # (2) Fisher 信息矩阵部分
        # ----------------------------------------------------------
        if fisher is not None:
            self.fisher = {k: v.detach().cpu().float() for k, v in fisher.items()}
            print(f"[EWC] Fisher matrix loaded with {len(self.fisher)} entries.")
        else:
            self.fisher = None
            print("[EWC] No Fisher matrix provided — no EWC regularization will be applied.")


    def ewc_loss(self):
        """
        返回“原始 EWC 值”（未乘 lambda_ewc），在 fp32 中计算。
        这里做了两个稳定化：
        1) 显式 to(device)+float()，避免半精度参与
        2) 按参数规模做平均（/count），让尺度与模型大小无关
        """
        device = next(self.parameters()).device
        if self.fisher is None or self.old_params is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        total = torch.zeros((), device=device, dtype=torch.float32)
        count = 0

        # 关闭 autocast，确保 fp32
        import torch.cuda.amp as amp
        with amp.autocast(enabled=False):
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if n not in self.fisher or n not in self.old_params:
                    continue

                f = self.fisher[n].to(device=device, dtype=torch.float32)
                d = (p.float() - self.old_params[n].to(device).float())
                total = total + (f * d.pow(2)).sum()
                count += d.numel()

            if count > 0:
                total = total / count

        # [DEBUG]
        if self.global_step % 1000 == 0:
             print(f"[EWC DEBUG] Step {self.global_step}")
             if self.fisher is None: print("  -> self.fisher is None")
             if self.old_params is None: print("  -> self.old_params is None")
             if self.fisher and self.old_params:
                 keys_f = set(self.fisher.keys())
                 keys_o = set(self.old_params.keys())
                 common = keys_f.intersection(keys_o)
                 print(f"  -> Common keys: {len(common)}")
                 if len(common) > 0:
                     k = list(common)[0]
                     p = dict(self.named_parameters())[k]
                     p_old = self.old_params[k].to(device)
                     diff = (p - p_old).norm()
                     print(f"  -> Sample param '{k}' diff: {diff.item()}")
                     print(f"  -> Count accumulated: {count}")
                     print(f"  -> Total EWC calculated: {total.item()}")

        return total  # 注意：这里不乘 lambda
    
    def accumulate_loss(self, loss_map, agent_pos):
        """
        loss_map: (mask_size, mask_size)  局部loss
        agent_pos: (y, x) 智能体在全局地图中的位置
        """
        ay, ax = agent_pos
        half = self.mask_size // 2

        for dy in range(self.mask_size):
            for dx in range(self.mask_size):
                global_y = ay + (dy - half)
                global_x = ax + (dx - half)

                # 边界检查，防止越界
                if 0 <= global_y < self.row and 0 <= global_x < self.col:
                    value = loss_map[dy, dx].item()
                    self.loss_accumulator[global_y][global_x].append(value)

    def compute_cell_loss(self, next_pred, next_true):
        # 计算每个位置的误差
        if self.env_type == 'crafter':
            # Classification loss per cell (with target clamping)
            loss_map = crafter_classification_loss(
                next_pred, next_true, reduction='none'
            )  # (B, H, W)
        else:
            # Standard Regression error
            error = torch.abs(next_pred - next_true)
            loss_map = error.mean(dim=1)  # (B, H, W)

        return loss_map



    def forward(self, state, action, info, inv=None):
        out = self.model(state, action, info, inv=inv)
        if len(out) == 3:
            next_state_pred, attentionWeight, inv_pred = out
            return next_state_pred, attentionWeight, inv_pred
        else:
            # Fallback for MLP or older models that only return 2 items
            next_state_pred, attentionWeight = out
            return next_state_pred, attentionWeight, None


    def loss_function(self, next_observations_predict, next_observations_true):
        loss_obs = self.loss(next_observations_predict.flatten(1), next_observations_true.flatten(1))
        loss = {'loss_obs':loss_obs}
        return loss
    

    def loss_function_weight(self, next_observations_predict, next_observations_true, obs_masked=None, obs_prev=None):
        """
        Custom Loss with tiered aggressive weighting:
        - Static: 1.0
        - Movement: +10.0
        - State Change (Door/Key Interaction): +100.0! (Critical for UED)
        """
        device = next_observations_predict.device 
        
        # 1. Base Loss (MSE for MiniGrid, CrossEntropy for Crafter)
        if self.env_type == 'crafter':
            # Classification loss from crafter_support (with target clamping)
            raw_error_map = crafter_classification_loss(
                next_observations_predict, next_observations_true, reduction='none'
            )  # (B, H, W)
        else:
            # MiniGrid: Standard Regression
            raw_sq_error = (next_observations_predict - next_observations_true) ** 2
            raw_error_map = raw_sq_error.mean(dim=1)  # (B, H, W)

             # 2. Change Mask (Any change)
        if obs_prev is not None:
             # Ensure types match
             if obs_prev.dtype != next_observations_true.dtype:
                 obs_prev = obs_prev.float() 
             
             # If stacked, only compare against the latest frame in the stack
             C_target = next_observations_true.size(1)
             C_prev = obs_prev.size(1)
             obs_prev_latest = obs_prev[:, -C_target:] if C_prev > C_target else obs_prev
             
             # General Diff (Sum of all channels)
             diff = torch.abs(next_observations_true - obs_prev_latest).sum(dim=1, keepdim=True)
             change_mask = (diff > 1e-5).float()
             
             # 3. State Channel Diff (Channel 2 is State: 0=Open, 1=Closed, 2=Locked)
             if C_target > 2:
                  # Focus strictly on channel 2 of the most recent frame
                  state_diff = torch.abs(next_observations_true[:, 2:3, :, :] - obs_prev_latest[:, 2:3, :, :])
                  state_change_mask = (state_diff > 1e-5).float()
             else:
                  state_change_mask = torch.zeros_like(change_mask)
             
        else:
             # Fallback if no prev frame (rare)
             change_mask = (next_observations_true.abs() > 1e-6).any(dim=1, keepdim=True).float()
             state_change_mask = torch.zeros_like(change_mask)

        # 4. Combine Weights
        # Base=1.0, Move=+10, Interaction=+100 -> Total 111.0 for Opening Door
        weights = 1.0 + (change_mask * 10.0) + (state_change_mask * 100.0)
        
        if obs_masked is not None:
             if obs_masked.ndim == 3:
                 static_mask = obs_masked.unsqueeze(1).float()
             else:
                 static_mask = obs_masked.float()
                 
             # INTERACTION = CHANGE * ELEMENT
             interaction_mask = change_mask * static_mask
             
             if self.env_type == 'crafter':
                 # For Crafter, these elements are stochastic entities (zombies, animals).
                 # We want to learn them but not let them dominate the loss due to random jitter.
                 stochastic_weight = 5.0
                 weights = weights + (interaction_mask * stochastic_weight)
             else:
                 # Original MiniGrid behavior: boost critical interactions like keys/doors
                 weights = weights + (interaction_mask * 100.0)

        # 5. Weighted Mean
        # Ensure raw_error_map and weights have matching dimensions
        if weights.ndim > raw_error_map.ndim:
             weights = weights.squeeze(1) # [B, 1, H, W] -> [B, H, W]
        
        loss = (raw_error_map * weights).mean()
        
        return {"loss_obs": loss}



    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.weight_decay)
        # reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'avg_val_loss_wm',
                "frequency": 1
            },
        }

    def preprocess_batch(self, batch, training=False):
        '''
        Preprocess the batch data: extract masked observations and object positions.
        batch['obs']: (B, C, H, W)
        '''
        obs = batch['obs']
        act = batch['act']
        obs_next = batch['obs_next']
        if self.env_type == 'with_obj':
            info = batch['info']
        else:
            info = None
        
        inv = batch.get('inv', None)
        inv_next = batch.get('inv_next', None)

        player_id = 13 if self.env_type == 'crafter' else 10
        agent_postion_yx_batch = minigrid_utils.get_agent_position(obs, player_id=player_id)
        obs_masked = minigrid_utils.extract_masked_state(obs, self.mask_size, agent_postion_yx_batch)
        obs_next_masked = minigrid_utils.extract_masked_state(obs_next, self.mask_size, agent_postion_yx_batch)

        # extract positions where objects are located (use the most recent frame if stacked)
        C_base = 2 if self.env_type == 'crafter' else 3
        curr_obj_idx = (self.frame_stack - 1) * C_base
        object_map = obs_masked[:, curr_obj_idx]  # 取最后一帧的第0通道 (B,H,W)
        if self.env_type == 'crafter':
            # Interactive elements in Crafter: cow(11), zombie(12), skeleton(13), arrow(14), plant(15)
            elements_mask = (object_map >= 11) & (object_map <= 15)
        else:
            key_mask = (object_map == 5)
            door_mask = (object_map == 4)
            lava_mask = (object_map == 9)
            elements_mask = key_mask | door_mask | lava_mask  # (B,H,W)
        
        ## visualization is now moved to training_step/validation_step for logits access
        self.step_counter += 1
        return obs_masked, act, obs_next_masked, info, elements_mask, agent_postion_yx_batch, inv, inv_next


    def training_step(self, batch, batch_idx):
        # —— 前向 & 主损失 —— #
        obs, act, obs_next, info, elements_mask, agent_pos, inv, inv_next = self.preprocess_batch(batch, True)
        obs_pred, attentionWeight, inv_pred = self(obs, act, info, inv=inv)

        # [NEW] Crafter WM Visualization (Only in LAST EPOCH to save time)
        is_last_epoch = False
        try:
            is_last_epoch = (self.current_epoch == self.trainer.max_epochs - 1)
        except:
            pass
            
        if self.visualizationFlag and is_last_epoch and (self.step_counter % self.visualize_every == 0):
            if self.env_type == 'crafter':
                visualize_crafter_wm(obs, obs_next, obs_pred, int(act[0].item()), self.step_counter, 
                                     save_dir=os.path.join("modelBased/log", "wm_visual/train"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_pos[0],
                                     inv=inv[0].cpu().numpy() if inv is not None else None,
                                     inv_next=inv_next[0].cpu().numpy() if inv_next is not None else None)
            else:
                # Fallback to legacy visualize_data for MiniGrid (requires whole map)
                # Note: this part needs whole map, which we have in 'batch'
                self.visual_func.visualize_data(batch['obs'], batch['obs'] + batch['obs_next'], act, obs, obs_next, info, self.step_counter, agent_pos)

        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        
        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # [TRAINING] Aggressive Weighted Loss for Optimization
        loss_dict = self.loss_function_weight(obs_pred, obs_next, elements_mask, obs_prev=obs)
        weighted_obs_loss = loss_dict['loss_obs']
        
        # [LOGGING] Metrics for Scientific Reporting
        if self.env_type == 'crafter':
            obs_pred_reconst = crafter_reconstruct_from_logits(obs_pred)
            raw_mse = F.mse_loss(obs_pred_reconst, obs_next)
            # Raw (unweighted) CE loss — the true training signal for Crafter
            from domain.crafter.crafter_support import crafter_classification_loss
            raw_ce = crafter_classification_loss(obs_pred, obs_next, reduction='mean')
        else:
            raw_mse = F.mse_loss(obs_pred, obs_next)
            raw_ce = None
        
        # —— raw EWC（未乘 λ）—— #
        ewc_raw = self.ewc_loss()

        # —— 快控制：这里用 Weighted Loss 来平衡 EWC，因为它们都在 "Optimization Space" —— #
        self.update_lambda_ewc_by_ratio(weighted_obs_loss, ewc_raw, self.ewc_ratio)

        # —— 合成总损失 —— #
        ewc_term = self.lambda_ewc * ewc_raw
        loss_total = weighted_obs_loss + ewc_term
        
        # Add inventory MSE loss dynamically if applicable
        if self.env_type == 'crafter' and inv_pred is not None and inv_next is not None:
            if inv is not None:
                inv_diff = torch.abs(inv_next - inv).float()
                inv_weights = 1.0 + (inv_diff > 1e-5).float() * 100.0
                inv_loss = (F.mse_loss(inv_pred, inv_next.float(), reduction='none') * inv_weights).mean()
            else:
                inv_loss = F.mse_loss(inv_pred, inv_next.float())
                
            loss_total = loss_total + 10.0 * inv_loss
            self.log("train/inv_loss", inv_loss, prog_bar=True, on_step=True, on_epoch=True)

        # —— 统一日志 —— #
        # Log raw_mse as 'loss_obs' for compatibility
        self.log("loss_obs", raw_mse, prog_bar=True, on_step=True, on_epoch=True)
        
        if raw_ce is not None:
            # Crafter: also log raw CE separately for clear monitoring
            self.log("train/ce_loss", raw_ce, prog_bar=True, on_step=True, on_epoch=True)
        
        # Log weighted loss separately for debugging
        self.log("train/loss_weighted", weighted_obs_loss, on_step=True, on_epoch=True)
        
        self.log("train/ewc_raw", ewc_raw.detach(), on_step=True, on_epoch=True)
        self.log("train/lambda_ewc", torch.tensor(self.lambda_ewc, device=obs.device),
                on_step=True, on_epoch=True)
        self.log("train/ewc_term", ewc_term.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_total", loss_total.detach(), on_step=True, on_epoch=True)

        if self.global_step % 1000 == 0:
            ce_str = f", CE: {raw_ce.item():.6f}" if raw_ce is not None else ""
            print(f"[Step {self.global_step}] "
                f"Raw MSE: {raw_mse.item():.6f}, "
                f"Weighted Loss: {weighted_obs_loss.item():.6f}"
                f"{ce_str}, "
                f"EWC: {ewc_raw.item():.6f}, "
                f"Total: {loss_total.item():.6f}")

        return loss_total

    
   
    def validation_step(self, batch, batch_idx):
        obs, act, obs_next, info, elements_mask, agent_position, inv, inv_next = self.preprocess_batch(batch)
        obs_pred, attention_weight, inv_pred = self(obs, act, info, inv=inv)
        # if self.hparams.freeze_weight:
        #     diff = torch.abs(obs_pred - obs_next)  # (128, 3, 3, 3)
        #     max_diff_per_group, max_indices = diff.reshape(diff.shape[0], -1).max(dim=1)  
        #     mask = max_diff_per_group > 0.1
        #     indices = torch.nonzero(mask, as_tuple=True)[0]
        #     for idx in indices:
        #         flat_idx = max_indices[idx].item()
        #         pred_val = obs_pred[idx].reshape(-1)[flat_idx].item()
        #         true_val = obs_next[idx].reshape(-1)[flat_idx].item()
        #         print(f"索引 {idx.item()} 最大差值: {max_diff_per_group[idx].item():.4f}, "
        #             f"pred={pred_val:.4f}, true={true_val:.4f}")
        
        # 将loss映射回全局并保存到列表中
        if getattr(self.hparams, "keep_cell_loss", False):
            loss_map = self.compute_cell_loss(obs_pred, obs_next)
            batch_size = loss_map.shape[0]
            for i in range(batch_size):
                agent_pos = agent_position[i].tolist()  # (y, x)
                self.accumulate_loss(loss_map[i], agent_pos)
  
        if obs_next.dtype != obs_pred.dtype:
            obs_next = obs_next.float()
        
        # [NEW] Crafter WM Visualization for Validation (Dataset 2)
        if self.visualizationFlag and batch_idx == 0:
            if self.env_type == 'crafter':
                visualize_crafter_wm(obs, obs_next, obs_pred, int(act[0].item()), self.step_counter, 
                                     save_dir=os.path.join("modelBased/log", "wm_visual/val"),
                                     full_map_size=batch['obs'].shape[-2:],
                                     agent_pos=agent_position[0],
                                     inv=batch['inv'][0].cpu().numpy() if 'inv' in batch else None,
                                     inv_next=batch['inv_next'][0].cpu().numpy() if 'inv_next' in batch else None)

        # Ensure obs is float for diff calculation
        if obs.dtype != obs_pred.dtype:
            obs = obs.float()

        # [VALIDATION METRIC]
        # Crafter → CE is the correct metric (discrete IDs, not numeric)
        # MiniGrid → MSE as before
        if self.env_type == 'crafter':
            from domain.crafter.crafter_support import crafter_classification_loss
            val_ce = crafter_classification_loss(obs_pred, obs_next, reduction='mean')
            loss_val = val_ce
            
            if inv_pred is not None and inv_next is not None:
                val_inv_loss = F.mse_loss(inv_pred, inv_next.float())
                loss_val = loss_val + val_inv_loss
                self.log("val/inv_loss", val_inv_loss, on_step=False, on_epoch=True)

            self.log("val/ce_loss", val_ce, on_step=False, on_epoch=True)
        else:
            raw_mse = F.mse_loss(obs_pred, obs_next)
            loss_val = raw_mse

        self.log("val_loss", loss_val, prog_bar=True)

        return {
            "loss_wm_val": loss_val,             
        }

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:

        if getattr(self.hparams, "keep_cell_loss", False):
            avg_loss_map = torch.zeros((self.row, self.col), device=self.device)
            for y in range(self.row):
                for x in range(self.col):
                    vals = self.loss_accumulator[y][x]
                    avg_loss_map[y, x] = sum(vals) / len(vals) if vals else 0

            self.loss_map_result = avg_loss_map.cpu().numpy()
        # 保存为 CSV（不包含 index）
        # df.to_csv("validation_21*21_emb_mask5.csv", index=False, header=False)
        # 绘制 loss 变化曲线
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 5))
        # plt.plot(batch_indices, losses, marker="o", linestyle="-")
        # plt.xlabel("Batch Index")
        # plt.ylabel("Validation Loss")
        # plt.title("Validation Loss per Batch")
        # plt.grid(True)

        avg_loss = torch.stack([x["loss_wm_val"] for x in outputs]).mean()
        self.log("avg_val_loss_wm", avg_loss)
        
        return {
            "avg_val_loss_wm": avg_loss,
        }

    def on_save_checkpoint(self, checkpoint):
        # Example checkpoint customization: removing specific keys if needed
        t = checkpoint['state_dict']
        pass  # No specific filtering needed for a simple NN

    def calc_loss(self, trajectory_data):
        """
        Compute loss for Learning Progress (LP) reward calculation.
        Args:
            trajectory_data: dict with 'obs', 'act', 'obs_next', 'info' keys, containing tensors.
        Returns:
            loss: scalar tensor
        """
        device = self.device
        batch = {
            'obs': trajectory_data['obs'].to(device),
            'act': trajectory_data['act'].to(device),
            'obs_next': trajectory_data['obs_next'].to(device),
            'info': trajectory_data['info']
        }
        
        if self.env_type != 'with_obj':
             batch['info'] = None

        obs_masked, act, obs_next_masked, info, elements_mask, _, inv, inv_next = self.preprocess_batch(batch, training=False)
        
        obs_pred, _, _ = self(obs_masked, act, info, inv=inv)
        
        if obs_next_masked.dtype != obs_pred.dtype:
            obs_next_masked = obs_next_masked.float()
            
        loss_dict = self.loss_function_weight(obs_pred, obs_next_masked, elements_mask)
        return loss_dict

    def on_train_end(self):
        """Save a final visualization at the end of training."""
        if self.visualizationFlag and self.env_type == 'crafter':
            # We don't easily have the last batch here, but we can signal or just rely on the last training_step/val_step
            # For now, we prints a message to confirm.
            print(f"[WM] Training ended. Final visualizations saved in modelBased/log/wm_visual/")




   


