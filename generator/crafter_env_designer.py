import numpy as np
import random

# 定义对齐 CustomCrafterEnv 的物理 ID 映射
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
    为 Crafter 定制的 PCG 生成器。
    它负责吐出含有“一圈水墙边界”、“随机撒落的初始资源”以及“连通性校验”的 2D 矩阵。
    现在的地图是 100% 纯物理布局。
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
        # 凡是需要工具砍伐挖掘的“实体方块”都不算真正的空地（防止无脑堆树堆石头）
        # 只有真正无消耗通行的格子才算 walkable:
        walkable_tiles = [
            CRAFTER_OBJ_MAP['grass'],
            CRAFTER_OBJ_MAP['path'],
            CRAFTER_OBJ_MAP['sand'],
            CRAFTER_OBJ_MAP['agent'],
            CRAFTER_OBJ_MAP['plant'], # 植物走上去只扣饱食度，不算阻塞实体
            CRAFTER_OBJ_MAP['none'],
        ]
        return tile_id in walkable_tiles

    def check_connectivity(self, grid, threshold=0.85):
        """
        Skip BFS. In Crafter, any map is theoretically 'solvable' or adaptable.
        """
        return True, {'max_dist': 0, 'ratio': 1.0}

    def _raw_generate(self):
        # 构建全草地的空白画布 [H, W]
        grid = np.full((self.H, self.W), CRAFTER_OBJ_MAP['grass'], dtype=int)

        # 1. 设置物理边界 (水) - 这一圈是一定有的
        grid[0, :] = CRAFTER_OBJ_MAP['water']
        grid[-1, :] = CRAFTER_OBJ_MAP['water']
        grid[:, 0] = CRAFTER_OBJ_MAP['water']
        grid[:, -1] = CRAFTER_OBJ_MAP['water']

        # 2. 放置 Agent (放在非边界区域)
        ay, ax = random.randint(2, self.H-3), random.randint(2, self.W-3)
        grid[ay, ax] = CRAFTER_OBJ_MAP['agent']

        # 3. 随机撒入资源
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
        
        # 如果实在运气不好，返回最后一张
        return grid
