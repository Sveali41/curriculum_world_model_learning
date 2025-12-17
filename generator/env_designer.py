'''
pcg_seeder
: A module for generating empty environments structure by inputting a seed value.
- Ensures that the generated environment is solvable.

Map editor
: A module for editing and customizing generated environments by applying the change from generator.
- applying a change mask to the base layout.

task_placer
: A module for placing starting_pos and goals within the generated environments.
- Ensures that the placement adheres to specified constraints.


'''
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Union


# 定义常量
WALL  = OBJECT_TO_IDX['wall']
FLOOR = OBJECT_TO_IDX['empty']
DOOR  = OBJECT_TO_IDX['door']
KEY   = OBJECT_TO_IDX['key']
LAVA  = OBJECT_TO_IDX['lava']

RED    = COLOR_TO_IDX['red']
YELLOW = COLOR_TO_IDX['yellow']
BLUE   = COLOR_TO_IDX['blue']
GREY   = COLOR_TO_IDX['grey']

# --- 核心映射表 ---
# action map for generator agent output
# 格式: Action_ID: (Type, Color)
ACTION_MAP = {
    # 0: Keep (特殊处理)
    1: (WALL, GREY),
    2: (FLOOR, 0),
    3: (LAVA, RED),
    
    4: (DOOR, RED),
    5: (DOOR, YELLOW),
    6: (DOOR, BLUE),
    
    7: (KEY, RED),
    8: (KEY, YELLOW),
    9: (KEY, BLUE)
}

def map_editor(base_map_obj, base_map_col, action_grid):
    """
    base_map_obj: the base object map from PCGSeeder
    base_map_col: the base color map from PCGSeeder
    action_grid:  Generator agent output action grid
    """
    new_obj = base_map_obj.copy()
    new_col = base_map_col.copy()
    
    # 遍历除了 0 (Keep) 以外的所有动作
    for act_id, (type_val, color_val) in ACTION_MAP.items():
        # 找到执行该动作的所有位置
        mask = (action_grid == act_id)
        
        # 统一修改
        new_obj[mask] = type_val
        new_col[mask] = color_val
        
    return new_obj, new_col


class PCGSeeder:
    """
    Procedural Content Generation Seeder for W/E canvas.
    Ensures:
    - global connectivity
    - sufficient empty space for element placement
    - reproducibility
    """

    def __init__(
        self,
        height: int,
        width: int,
        min_empty_ratio: Union[float, Tuple[float, float]] = 0.3,
        max_tries: int = 100,
        structure_mode: str = "pure_random",  # or "blob"
    ):
        self.H = height
        self.W = width
        self.max_tries = max_tries
        self.structure_mode = structure_mode

        if isinstance(min_empty_ratio, tuple):
            self.min_empty_range = min_empty_ratio
        else:
            self.min_empty_range = (min_empty_ratio, min_empty_ratio)


    def generate(
        self,
        z: Optional[int] = None,
        return_info: bool = False
    ):
        if z is not None:
            random.seed(z)
            np.random.seed(z)

        N = self.H * self.W
        min_ratio = random.uniform(*self.min_empty_range)
        min_empty = int(min_ratio * N)

        tries = 0

        for _ in range(self.max_tries):
            tries += 1

            if self.structure_mode == "blob":
                grid = self._sample_blob_canvas()
            else:
                grid = self._sample_random_canvas()

            empty_count = np.sum(grid == FLOOR)
            if empty_count < min_empty:
                continue

            if self._is_fully_connected(grid):
                if return_info:
                    return grid, {
                        "tries": tries,
                        "empty_ratio": empty_count / N,
                        "fallback": False,
                        "structure_mode": self.structure_mode,
                    }
                return grid

        # ---------- fallback ----------
        fallback_grid = self._fallback_canvas()

        if return_info:
            return fallback_grid, {
                "tries": tries,
                "empty_ratio": np.mean(fallback_grid == FLOOR),
                "fallback": True,
                "structure_mode": "fallback",
            }

        return fallback_grid


    def _sample_random_canvas(self) -> np.ndarray:
        grid = np.ones((self.H, self.W), dtype=int)

        for i in range(self.H):
            for j in range(self.W):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    grid[i, j] = WALL
                else:
                    grid[i, j] = (
                        WALL
                        if random.random() < random.random()
                        else FLOOR
                    )
        return grid

    def _sample_blob_canvas(self) -> np.ndarray:
        grid = np.full((self.H, self.W), WALL, dtype=int)

        y, x = self.H // 2, self.W // 2
        grid[y, x] = FLOOR

        steps = (self.H * self.W) // 2
        for _ in range(steps):
            dy, dx = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            y = max(1, min(self.H - 2, y + dy))
            x = max(1, min(self.W - 2, x + dx))
            grid[y, x] = FLOOR

        return grid


    def _is_fully_connected(self, grid: np.ndarray) -> bool:
        empties = np.argwhere(grid == FLOOR)
        if len(empties) == 0:
            return False

        start = tuple(empties[0])
        visited = {start}
        queue = deque([start])

        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < self.H
                    and 0 <= nx < self.W
                    and grid[ny, nx] == FLOOR
                    and (ny, nx) not in visited
                ):
                    visited.add((ny, nx))
                    queue.append((ny, nx))

        return len(visited) == len(empties)


    def _fallback_canvas(self) -> np.ndarray:
        pool = self._generate_fallback_pool()
        return random.choice(pool)

    def _generate_fallback_pool(self):
        raw_pool = []

        # 1. Pure empty room
        raw_pool.append(self._fallback_empty_room())

        # 2. Empty room + sparse pillars
        raw_pool.append(self._fallback_room_with_pillars(0.05))
        raw_pool.append(self._fallback_room_with_pillars(0.1))
        raw_pool.append(self._fallback_room_with_pillars(0.15))

        # 3. Blob (still 2D, not corridor)
        raw_pool.append(self._sample_blob_canvas())

        # -------- filter by empty_ratio --------
        min_r, max_r = self.min_empty_range
        filtered = [
            g for g in raw_pool
            if min_r <= np.mean(g == FLOOR) <= max_r
        ]

        # soft fallback: at least satisfy min_r
        if len(filtered) == 0:
            filtered = [
                g for g in raw_pool
                if np.mean(g == FLOOR) >= min_r
            ]

        # last resort (should not happen)
        if len(filtered) == 0:
            filtered = raw_pool

        return filtered

    def _fallback_empty_room(self):
        grid = np.full((self.H, self.W), FLOOR, dtype=int)
        grid[0, :] = grid[-1, :] = WALL
        grid[:, 0] = grid[:, -1] = WALL
        return grid

    def _fallback_room_with_pillars(self, pillar_ratio=0.1):
        """
        Large open room with sparse wall pillars.
        Guarantees many valid placement positions.
        """
        grid = np.full((self.H, self.W), FLOOR, dtype=int)

        grid[0, :] = grid[-1, :] = WALL
        grid[:, 0] = grid[:, -1] = WALL

        candidates = [
            (i, j)
            for i in range(1, self.H - 1)
            for j in range(1, self.W - 1)
        ]

        random.shuffle(candidates)
        num_pillars = int(len(candidates) * pillar_ratio)

        for y, x in candidates[:num_pillars]:
            grid[y, x] = WALL

        return grid



if __name__ == "__main__":
    seeder = PCGSeeder(
        height=10,
        width=10,
        min_empty_ratio=(0.3, 0.9),
        max_tries=3000,
        structure_mode="pure_random",
    )

    grid, info = seeder.generate(z=3, return_info=True)
    print(grid)
    print(info)
