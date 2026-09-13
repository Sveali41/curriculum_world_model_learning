"""Create three small, solvable 7x7 MiniGrid maps for fast explorer tests."""

from pathlib import Path

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

from modelBased.exploration.minigrid_corpus import MiniGridCorpusWriter, load_corpus, reachable_positions


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "trainer/data/minigrid/explorer_ab/mac_3_maps_7x7_seed0.npz"


def _map(rows, *, key_colors, locked_colors, closed_colors=None):
    if len(rows) != 7 or any(len(row) != 7 for row in rows):
        raise ValueError("all maps must be 7x7")
    closed_colors = {} if closed_colors is None else closed_colors
    obj = np.full((7, 7), OBJECT_TO_IDX["empty"], dtype=np.int64)
    color = np.full((7, 7), COLOR_TO_IDX["grey"], dtype=np.int64)
    state = np.zeros((7, 7), dtype=np.int64)
    symbols = {"#": "wall", ".": "empty", "S": "agent", "G": "goal", "L": "lava"}
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol in symbols:
                obj[y, x] = OBJECT_TO_IDX[symbols[symbol]]
            elif symbol in {"K", "D", "O"}:
                obj[y, x] = OBJECT_TO_IDX["key" if symbol == "K" else "door"]
            else:
                raise ValueError(f"unknown map symbol {symbol!r}")
    for (y, x), name in key_colors.items():
        color[y, x] = COLOR_TO_IDX[name]
    for (y, x), name in locked_colors.items():
        color[y, x] = COLOR_TO_IDX[name]
        state[y, x] = STATE_TO_IDX["locked"]
    for (y, x), name in closed_colors.items():
        color[y, x] = COLOR_TO_IDX[name]
        state[y, x] = STATE_TO_IDX["closed"]
    return obj, color, state


def main():
    maps = [
        _map(
            ["#######", "#S..K.#", "#.##D##", "#....G#", "#.L...#", "#.....#", "#######"],
            key_colors={(1, 4): "blue"},
            locked_colors={(2, 4): "blue"},
        ),
        _map(
            ["#######", "#S...K#", "#.###D#", "#...G.#", "#..O..#", "#..L..#", "#######"],
            key_colors={(1, 5): "red"},
            locked_colors={(2, 5): "red"},
            closed_colors={(4, 3): "green"},
        ),
        _map(
            ["#######", "#S.K..#", "#.#D..#", "#...G.#", "#O..L.#", "#.....#", "#######"],
            key_colors={(1, 3): "green"},
            locked_colors={(2, 3): "green"},
            closed_colors={(4, 1): "red"},
        ),
    ]
    object_maps = np.stack([item[0] for item in maps])
    color_maps = np.stack([item[1] for item in maps])
    state_maps = np.stack([item[2] for item in maps])
    writer = MiniGridCorpusWriter(
        OUTPUT,
        expected_size=3,
        generation_seed=0,
        generation_metadata={"purpose": "fast_effect_reward_7x7", "map_size": "7x7"},
    )
    writer.append_batch(
        object_maps=object_maps,
        color_maps=color_maps,
        state_maps=state_maps,
        inventory_tokens=np.zeros(3, dtype=np.int64),
        iteration=0,
        start_dirs=np.zeros(3, dtype=np.int64),
    )
    path = writer.finalize()
    corpus = load_corpus(path, expected_size=3)
    for index in range(3):
        reachable = reachable_positions(
            corpus.object_maps[index], corpus.color_maps[index], corpus.state_maps[index], 0
        )
        goal = tuple(np.argwhere(corpus.object_maps[index] == OBJECT_TO_IDX["goal"])[0])
        if goal not in reachable:
            raise RuntimeError(f"map {index} goal is not reachable")
        print(f"map={index} reachable={len(reachable)} goal={goal}")
    print(path)


if __name__ == "__main__":
    main()
