
import numpy as np
import torch

class RandomGeneratorAgent:
    def __init__(self, num_actions, device='cuda'):
        self.num_actions = num_actions
        self.device = device
        # Dummy internal state to match PPO interface if needed
        self.last_mean_reward = 0.0 
    
    def select_action(self, base_map, prev_data, mask, max_edits):
        """
        Return random actions for the batch.
        Signature matches GeneratorPPO.select_action:
        inputs:
            base_map: [B, C, H, W]
            prev_data: (prev_map, prev_heat) or None
            mask: [B, 1, H, W] (immutable mask, 1.0=immutable)
            max_edits: float ratio
        
        returns:
            action: [B, H, W]  <-- CRITICAL: Must be spatial
            logprob: [B]
            value: [B]
            topk_mask: [B, num_actions] (dummy)
            global_ctx: [1, context_dim] (dummy)
        """
        B, C, H, W = base_map.shape
        
        # Random spatial actions [B, H, W]
        # Actions are 0..7
        action = torch.randint(0, self.num_actions, (B, H, W), device=self.device)
        
        # --- [MODIFIED] Enforce max_edits constraint for DR ---
        # Select K random cells to edit
        num_cells = H * W
        
        # [NEW] Sample edit ratio uniformly from [0, max_edits]
        # This makes max_edits a strict UPPER BOUND, not a fixed target.
        # k will vary per sample in the batch for diversity.
        actual_ratio = torch.rand((B, 1), device=self.device) * max_edits
        k_batch = (actual_ratio * num_cells).long()
        
        # Since topk requires a single K, we take the max K in the batch for the TopK op, 
        # and then mask out the extras later. Or simpler: just use one random ratio for the whole batch?
        # Let's use one random ratio for the whole batch for simplicity and efficiency.
        random_ratio = np.random.uniform(0.0, max_edits)
        k = int(round(random_ratio * num_cells))
        k = max(1, k) # Ensure at least 1 edit if ratio > 0, else 0
        k = min(k, num_cells)
        
        # Create random mask for K edits
        # Random noise for sorting
        rand_noise = torch.rand((B, num_cells), device=self.device)
        # Top-K indices
        _, indices = torch.topk(rand_noise, k, dim=1)
        
        # Scatter to mask [B, H*W] -> [B, H, W]
        edit_mask = torch.zeros((B, num_cells), dtype=torch.bool, device=self.device)
        edit_mask.scatter_(1, indices, True)
        edit_mask = edit_mask.view(B, H, W)
        
        # Apply constraint: Only edit where edit_mask is True
        # action[~edit_mask] = 0 (assuming 0 is No-Op/Empty)
        # Note: action 0 might change something if base is not empty. 
        # But usually action 0 is "No-Op" or "Empty".
        # If we want to strictly "Not Edit", we should output action=0 (if 0 is 'Keep').
        # Actually random agent generates 'actions'. Action 0 usually means "Floor/Empty" or "No-Op".
        # Let's assume action 0 is safe default for "Do Nothing" or "Empty".
        action[~edit_mask] = 0

        # Optional: Apply mask to set action=0 where immutable (mask=1.0)
        # mask is [B, 1, H, W]
        if mask is not None:
             # Expand mask to [B, H, W]
             m = mask.squeeze(1) > 0.5
             action[m] = 0 # No-op on immutable
             
        # Dummy logp, value
        logprob = torch.zeros(B, device=self.device)
        value = torch.zeros(B, device=self.device)
        
        # Dummy topk_mask (all valid)
        topk_mask = torch.ones((B, self.num_actions), device=self.device)
        
        # Dummy global_ctx
        # Need to know context_dim? Usually 64. 
        # But we can just return a zero tensor of shape [1, 64] 
        # Or better, don't hardcode 64 if possible, but PPO uses self.context_dim.
        # Let's use a safe default 64 or 1.
        global_ctx = torch.zeros((1, 64), device=self.device)

        return action, logprob, value, topk_mask, global_ctx

    def update(self, *args, **kwargs):
        """No-op update for random agent"""
        return 0.0 # gen_loss (scalar)
    
    def save_buffer(self, *args, **kwargs):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
