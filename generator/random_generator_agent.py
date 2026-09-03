
import numpy as np
import torch


class RandomGeneratorAgent:
    def __init__(
        self, num_actions, device='cuda', edit_action_group_sizes=None,
        env_type="minigrid",
    ):
        self.num_actions = num_actions
        self.device = device
        self.env_type = str(env_type).lower()
        group_sizes = np.asarray(
            edit_action_group_sizes
            if edit_action_group_sizes is not None
            else np.ones(max(num_actions - 1, 1)),
            dtype=np.float64,
        )
        if group_sizes.shape != (num_actions - 1,) or np.any(group_sizes <= 0):
            raise ValueError(
                "edit_action_group_sizes must contain one positive value per edit action"
            )
        self.edit_action_weights = torch.as_tensor(
            1.0 / group_sizes, dtype=torch.float32, device=device
        )
        # Dummy internal state to match PPO interface if needed
        self.last_mean_reward = 0.0 
    
    def select_action(self, base_map, prev_data, mask, 
                      max_edits_layout=0.1, max_stats_edit_ratio=0.1, stats_heat=None):
        """
        Return random actions for the batch.
        Signature matches GeneratorPPO.select_action:
        inputs:
            base_map: [B, C, H, W]
            prev_data: (prev_map, prev_heat) or None
            mask: [B, 1, H, W] (immutable mask, 1.0=immutable)
            max_edits_layout: expected per-cell terrain edit probability
            max_stats_edit_ratio: float ratio for inventory [0..1]
        
        returns:
            action: [B, H, W]  <-- Terrain actions
            stats_action: [B, 32] <-- Inventory actions (32 piano keys)
            logprob: [B]
            value: [B]
            topk_mask: [B, H, W] actual sampled edit mask
            location_order: [B, H, W] sampled order (diagnostic only)
            global_ctx: [1, context_dim] (placeholder)
        """
        B, C, H, W = base_map.shape
        
        # --- 1. Terrain Random Edits ---
        action = torch.zeros((B, H, W), dtype=torch.long, device=self.device)
        immutable = (
            mask.squeeze(1) > 0.5
            if mask is not None
            else torch.zeros((B, H, W), dtype=torch.bool, device=self.device)
        )
        edit_probability = float(np.clip(max_edits_layout, 0.0, 1.0))
        edit_mask = (torch.rand((B, H, W), device=self.device) < edit_probability)
        edit_mask &= ~immutable
        edit_count = int(edit_mask.sum().item())
        if edit_count > 0:
            action[edit_mask] = torch.multinomial(
                self.edit_action_weights,
                edit_count,
                replacement=True,
            ) + 1
             
        # --- 2. Inventory Random Edits ---
        if self.env_type == "minigrid":
            # One native carrying slot: 0=empty, 1..4=Y/R/B/G key.
            edit_inventory = torch.rand(B, device=self.device) < float(
                np.clip(max_stats_edit_ratio, 0.0, 1.0)
            )
            stats_action = torch.zeros((B, 1), dtype=torch.long, device=self.device)
            stats_action[edit_inventory, 0] = torch.randint(
                1, 5, (int(edit_inventory.sum()),), device=self.device
            )
            topk_stats_mask = edit_inventory.unsqueeze(1)
        else:
            # Crafter uses 32 piano keys: +1 and +5 for each of 16 slots.
            num_keys = 32
            current_p = np.random.uniform(0.0, max_stats_edit_ratio)
            rand_tensor = torch.rand((B, num_keys), device=self.device)
            stats_action = (rand_tensor < current_p).float()
            topk_stats_mask = torch.ones(
                (B, num_keys), device=self.device, dtype=torch.bool
            )
            
        # --- 3. Dummies ---
        logprob = torch.zeros(B, device=self.device)
        value = torch.zeros(B, device=self.device)
        topk_mask = edit_mask
        location_order = torch.zeros((B, H, W), device=self.device, dtype=torch.long)
        for batch_idx in range(B):
            positions = torch.nonzero(edit_mask[batch_idx].reshape(-1), as_tuple=False).flatten()
            if positions.numel() > 0:
                location_order[batch_idx].view(-1)[positions] = torch.arange(
                    1, positions.numel() + 1, device=self.device
                )
        global_ctx = torch.zeros((1, 64), device=self.device)

        return (
            action, stats_action, logprob, value, topk_mask,
            location_order, topk_stats_mask, global_ctx,
        )

    def update(self, *args, **kwargs):
        """No-op update for random agent"""
        return 0.0, 0.0 # gen_loss, entropy (scalars)
    
    def save_buffer(self, *args, **kwargs):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
