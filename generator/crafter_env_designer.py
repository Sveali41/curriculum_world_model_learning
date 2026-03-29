import numpy as np
import random

# 定义独立于真实引擎内部ID的生成器张量符号映射
# 生成器只理解0-10这些连续的数字，不理解复杂的真实游戏材质ID
CRAFTER_OBJ_MAP = {
    'grass': 0,
    'water': 1,
    'tree': 2,
    'stone': 3,
    'coal': 4,
    'iron': 5,
    'lava': 6,
    'zombie': 7,
    'table': 8,
    'furnace': 9,
    'agent': 10
}

CRAFTER_ACTION_MAP = {
    0: None, # Keep/No-op (由生成器底层处理)
    1: CRAFTER_OBJ_MAP['tree'],
    2: CRAFTER_OBJ_MAP['stone'],
    3: CRAFTER_OBJ_MAP['coal'],
    4: CRAFTER_OBJ_MAP['iron'],
    5: CRAFTER_OBJ_MAP['water'],
    6: CRAFTER_OBJ_MAP['table'],
    7: CRAFTER_OBJ_MAP['furnace'],
    # 背包装备区的专属操作 (由生成器特殊逻辑处理其落点)
    8: "inventory_add_1",
    9: "inventory_add_2",
}

class CrafterPCGSeeder:
    """
    为 Crafter 定制的 PCG 生成器。
    它负责吐出含有“一圈水墙边界”、“随机撒落的初始资源”、“中心主角”以及“底部一行隐藏背包”的 2D 矩阵。
    """
    def __init__(
        self,
        height: int,
        width: int,
        tree_ratio: float = 0.08,
        water_ratio: float = 0.02,
        stone_ratio: float = 0.00,
        structure_mode: str = "pure_random",
    ):
        # 这里的 height 和 width 是配置文件传来的。
        # 此处的 height = 物理地图高度 H + 1 (最后一行用来存背包配置)
        self.H = height
        self.W = width
        
        self.tree_ratio = tree_ratio
        self.water_ratio = water_ratio
        self.stone_ratio = stone_ratio
        self.structure_mode = structure_mode

    def generate(self, z: int = None, return_info: bool = False):
        if z is not None:
            random.seed(z)
            np.random.seed(z)

        # 构建全草地的空白画布 [H, W]
        grid = np.full((self.H, self.W), CRAFTER_OBJ_MAP['grass'], dtype=int)

        # 1. 设置物理边界 (水)
        # 注意：最后一行 (self.H - 1) 是控制面板，不在地形里！
        # 物理地图的最后一行是 self.H - 2
        phys_H = self.H - 1

        grid[0, :] = CRAFTER_OBJ_MAP['water']            # 顶边 
        grid[phys_H - 1, :] = CRAFTER_OBJ_MAP['water']   # 底边
        grid[:phys_H, 0] = CRAFTER_OBJ_MAP['water']      # 左边
        grid[:phys_H, -1] = CRAFTER_OBJ_MAP['water']     # 右边

        # 2. 放置 Agent (放在中心区域)
        center_y, center_x = phys_H // 2, self.W // 2
        grid[center_y, center_x] = CRAFTER_OBJ_MAP['agent']

        # 3. 随机撒入少量新手村资源 (避免 Cold Start)
        inner_coords = [
            (i, j)
            for i in range(1, phys_H - 1)
            for j in range(1, self.W - 1)
            if (i, j) != (center_y, center_x)
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

        # 4. 初始化控制面板 / 背包栏
        # 最后一行强制清零 (全是0)，确保初始背包纯净
        grid[self.H - 1, :] = 0

        if return_info:
            return grid, {"structure_mode": self.structure_mode}
        return grid
