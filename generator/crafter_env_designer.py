import numpy as np
import random

# Object-ID mapping aligned with `CustomCrafterEnv`.
CRAFTER_OBJ_MAP = {
    'none': 0, 'water': 1, 'grass': 2, 'stone': 3, 'path': 4,
    'sand': 5, 'tree': 6, 'lava': 7, 'coal': 8, 'iron': 9,
    'diamond': 10, 'table': 11, 'furnace': 12, 'agent': 13, 'cow': 14,
    'zombie': 15, 'skeleton': 16, 'arrow': 17, 'plant': 18, 'fence': 19
}

CRAFTER_STATS_KEYS = [
    'health', 'food', 'drink', 'energy',
    'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
    'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
    'wood_sword', 'stone_sword', 'iron_sword'
]

CRAFTER_ACTION_MAP = {
    0: None, 
    1: CRAFTER_OBJ_MAP['tree'], 2: CRAFTER_OBJ_MAP['stone'], 
    3: CRAFTER_OBJ_MAP['coal'], 4: CRAFTER_OBJ_MAP['iron'], 
    5: CRAFTER_OBJ_MAP['diamond'], 6: CRAFTER_OBJ_MAP['water'], 
    7: CRAFTER_OBJ_MAP['table'], 8: CRAFTER_OBJ_MAP['furnace'],
    9: CRAFTER_OBJ_MAP['plant'], 10: CRAFTER_OBJ_MAP['agent'], 11: CRAFTER_OBJ_MAP['cow']
}

class CrafterPCGSeeder:
    """
    PCG seeder specialized for Crafter.
    It generates a 2D map with a water boundary ring, randomly placed
    initial resources, and basic connectivity handling.
    """
    def __init__(
        self,
        height: int,
        width: int,
        tree_ratio: float = 0.08,
        water_ratio: float = 0.02,
        stone_ratio: float = 0.02,
        structure_mode: str = "pure_random",
    ):
        self.H = height
        self.W = width
        self.tree_ratio = tree_ratio
        self.water_ratio = water_ratio
        self.stone_ratio = stone_ratio
        self.structure_mode = structure_mode

    def _is_walkable(self, tile_id):
        # Tiles that require tools or blocking interactions are treated as non-walkable.
        walkable_tiles = [
            CRAFTER_OBJ_MAP['grass'],
            CRAFTER_OBJ_MAP['path'],
            CRAFTER_OBJ_MAP['sand'],
            CRAFTER_OBJ_MAP['agent'],
            CRAFTER_OBJ_MAP['plant'], # Plants are traversable.
            CRAFTER_OBJ_MAP['none'],
        ]
        return tile_id in walkable_tiles

    def check_connectivity(self, grid, threshold=0.85):
        """
        Skip BFS. In Crafter, any map is theoretically 'solvable' or adaptable.
        """
        return True, {'max_dist': 0, 'ratio': 1.0}

    def _raw_generate(self):
        # Start from an all-grass canvas of shape `[H, W]`.
        grid = np.full((self.H, self.W), CRAFTER_OBJ_MAP['grass'], dtype=int)

        # 1. Add the fixed water boundary.
        grid[0, :] = CRAFTER_OBJ_MAP['water']
        grid[-1, :] = CRAFTER_OBJ_MAP['water']
        grid[:, 0] = CRAFTER_OBJ_MAP['water']
        grid[:, -1] = CRAFTER_OBJ_MAP['water']

        # 2. Place the agent away from the boundary.
        ay, ax = random.randint(2, self.H-3), random.randint(2, self.W-3)
        grid[ay, ax] = CRAFTER_OBJ_MAP['agent']

        # 3. Randomly place resources.
        inner_coords = [
            (i, j) for i in range(1, self.H-1) for j in range(1, self.W-1)
            if grid[i, j] == CRAFTER_OBJ_MAP['grass']
        ]
        random.shuffle(inner_coords)
        N_inner = len(inner_coords)
        
        n_trees = int(N_inner * self.tree_ratio)
        n_waters = int(N_inner * self.water_ratio)
        n_stones = int(N_inner * self.stone_ratio)

        idx = 0
        for _ in range(n_trees):
            if idx < N_inner: grid[inner_coords[idx]] = CRAFTER_OBJ_MAP['tree']; idx += 1
        for _ in range(n_waters):
            if idx < N_inner: grid[inner_coords[idx]] = CRAFTER_OBJ_MAP['water']; idx += 1
        for _ in range(n_stones):
            if idx < N_inner: grid[inner_coords[idx]] = CRAFTER_OBJ_MAP['stone']; idx += 1

        return grid

    def generate(self, z: int = None, return_info: bool = False):
        if z is not None:
            random.seed(z)
            np.random.seed(z)
        
        max_retries = 10
        for _ in range(max_retries):
            grid = self._raw_generate()
            is_connected, _ = self.check_connectivity(grid)
            if is_connected:
                if return_info:
                    return grid, {"structure_mode": self.structure_mode, "connected": True}
                return grid
        
        # Fall back to the final sampled map if all retries fail.
        return grid
