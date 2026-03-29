
import numpy as np
import torch

class RandomGeneratorAgent:
    def __init__(self, num_actions, device='cuda'):
        self.num_actions = num_actions
        self.device = device
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
            max_edits_layout: float ratio for terrain
            max_stats_edit_ratio: float ratio for inventory [0..1]
        
        returns:
            action: [B, H, W]  <-- Terrain actions
            stats_action: [B, 32] <-- Inventory actions (32 piano keys)
            logprob: [B]
            value: [B]
            topk_mask: [B, num_actions] (dummy)
            global_ctx: [1, context_dim] (dummy)
        """
        B, C, H, W = base_map.shape
        num_cells = H * W
        
        # --- 1. Terrain Random Edits ---
        action = torch.randint(0, self.num_actions, (B, H, W), device=self.device)
        
        # Randomly choose K cells to edit based on ratio
        random_ratio = np.random.uniform(0.0, max_edits_layout)
        k = max(1, int(round(random_ratio * num_cells)))
        
        rand_noise = torch.rand((B, num_cells), device=self.device)
        _, indices = torch.topk(rand_noise, k, dim=1)
        
        edit_mask = torch.zeros((B, num_cells), dtype=torch.bool, device=self.device)
        edit_mask.scatter_(1, indices, True)
        edit_mask = edit_mask.view(B, H, W)
        
        action[~edit_mask] = 0 # No-op where not selected
        
        # Mask immutable cells
        if mask is not None:
             action[mask.squeeze(1) > 0.5] = 0
             
        # --- 2. Inventory Random Edits (32 Piano Keys) ---
        # Slots 0-15 (Key 0-15): Inc by 1
        # Slots 0-15 (Key 16-31): Inc by 5
        num_keys = 32
        stats_action = torch.zeros((B, num_keys), device=self.device)
        
        # How many keys to press?
        k_stats = max(1, int(round(max_stats_edit_ratio * num_keys)))
        
        for b in range(B):
            # Randomly pick indices to modify
            indices_stats = np.random.choice(num_keys, k_stats, replace=False)
            # Binary actions (0 or 1)
            stats_action[b, indices_stats] = 1.0
            
        # --- 3. Dummies ---
        logprob = torch.zeros(B, device=self.device)
        value = torch.zeros(B, device=self.device)
        topk_mask = torch.ones((B, self.num_actions), device=self.device)
        global_ctx = torch.zeros((1, 64), device=self.device)

        return action, stats_action, logprob, value, topk_mask, global_ctx

    def update(self, *args, **kwargs):
        """No-op update for random agent"""
        return 0.0, 0.0 # gen_loss, entropy (scalars)
    
    def save_buffer(self, *args, **kwargs):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
