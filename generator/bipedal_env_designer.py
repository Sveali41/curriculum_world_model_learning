import random
import numpy as np

# Map discrete actions (0-9) to obstacle types and sampling definitions.
# Each entry is a lambda returning the exact string token, applying a bounded noise.
def get_noisy_val(center_val, noise_range=0.3):
    return round(center_val + random.uniform(-noise_range, noise_range), 2)

ACTION_SAMPLERS = {
    0: lambda: ('G', random.randint(10, 15)),  # NO-OP / Long Grass (Randomized flat ground)
    1: lambda: ('S', get_noisy_val(1.0, 0.4)),    # Small Stump (0.6 ~ 1.4)
    2: lambda: ('S', get_noisy_val(2.0, 0.4)),    # Medium Stump (1.6 ~ 2.4)
    3: lambda: ('S', get_noisy_val(3.0, 0.4)),    # Large Stump (2.6 ~ 3.4)
    # 4~5: Pit obstacles. Values must remain integer-aligned.
    4: lambda: ('P', random.randint(1, 2)),       # Small Pit (Width 1~2)
    5: lambda: ('P', random.randint(3, 4)),       # Large Pit (Width 3~4)

    # 6~8: Stair obstacles. Values must remain integer step counts.
    6: lambda: ('T', random.randint(2, 3)),       # Low Stairs Up (2~3 steps)
    7: lambda: ('T', random.randint(4, 6)),       # High Stairs Up (4~6 steps)
    8: lambda: ('T', random.randint(-3, -2)),     # Low Stairs Down (2~3 steps down)
    9: lambda: ('R', get_noisy_val(2.5, 0.5)),    # Rough Terrain
}

# In BipedalWalker, the action map is practically identity-like because GeneratorInterface
# will directly store the action_id into the grid to be interpreted later.
# Expose a simple lookup so the PPO agent can infer the action-space size.
ACTION_TABLE_BIPEDAL = {i: i for i in range(10)}

class BipedalPCGSeeder:
    """
    A 1D seeder for BipedalWalker procedural environments.
    Initializes a 1xW canvas where the agent is placed at (0, 0).
    The map represents the obstacle layout choices along the track.
    """
    def __init__(self, width=5):
        self.width = width
        # Height is strictly 1 for string-based sequence problems
        self.height = 1

    def generate(self, batch_size):
        # Creates a batch of empty canvases (all 0s, meaning No-op/Grass)
        # Type must be compatible with the GeneratorInterface expected torch format 
        # (usually float/int mapping). We'll stick to int IDs.
        grid = np.zeros((batch_size, 1, self.height, self.width), dtype=np.int32)
        
        # We must define an 'agent' coordinate for GeneratorInterface compatibility,
        # but in BipedalWalker, the agent coordinates don't drive spatial dynamics within the grid.
        # We just set agent_poses to (0, 0)
        agent_poses = [(0, 0) for _ in range(batch_size)]
        
        return grid, agent_poses


def bipedal_array_to_layout_str(grid_1d_np, active_width=None):
    """
    Translates a 1D numpy array of action_ids into a valid BipedalWalkerCustom layout string.
    """
    parts = ["G20"]  # Initial spawn safe pad (increased to G20 for runway)

    action_ids = grid_1d_np.flatten()
    if active_width is not None:
        active_width = max(1, min(int(active_width), action_ids.shape[0]))
        action_ids = action_ids[:active_width]

    # 1D array of shape [W]
    for action_id in action_ids:
        a = int(action_id)
        if a == 0 or a not in ACTION_SAMPLERS:
            # Action 0 or invalid -> Safe grass pad with randomized length
            parts.append(f"G{random.randint(10, 15)}")
        else:
            sampler = ACTION_SAMPLERS[a]
            if callable(sampler):
                t, v = sampler()
            else:
                t, v = sampler
            parts.append(f"{t}{v}")
            
            # Post-obstacle spacing (Tighter gap)
            gap = random.randint(2, 5)                  # Tighter gap to match target tasks (was 3-7)
            parts.append(f"G{gap}")

    # Add a single long tail pad so only the beginning of the course carries edits.
    parts.append(f"G{random.randint(10, 15)}")

    return " ".join(parts)
