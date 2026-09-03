"""Run the preregistered memory-on/off explorer comparison on a frozen corpus."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
WM_ROOT = ROOT_DIR / "wm"
sys.path.insert(0, str(WM_ROOT))
sys.path.insert(1, str(ROOT_DIR))
os.environ.setdefault("PROJECT_ROOT", str(ROOT_DIR))
os.environ.setdefault("WM_ROOT", str(WM_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX
from omegaconf import OmegaConf

from domain.minigrid.action_codec import carrying_token_from_env, compact_to_native
from modelBased.exploration.count_based import MINIGRID_PLAYER_ID
from modelBased.exploration.minigrid_corpus import load_corpus, make_env, reachable_positions
from modelBased.exploration.minigrid_rmax import MiniGridRMaxExplorer


EXPLORER_SEEDS = (17, 29, 41)
VARIANTS = ("episodic", "none")
TRANSITION_BUDGET = 1000
ROLLOUT_STEPS = 256
BLOCK_SIZE = 80
EXPECTED_MAPS = 400
BOOTSTRAP_SAMPLES = 10_000
JOINT_ITERATION_MAPS = 8


def _image(observation):
    return observation["image"] if isinstance(observation, dict) else observation


def _position(image):
    positions = np.argwhere(np.asarray(image)[..., 0] == MINIGRID_PLAYER_ID)
    if len(positions) != 1:
        raise ValueError(f"expected exactly one observed agent, found {len(positions)}")
    y, x = (int(value) for value in positions[0])
    return y, x


def _door_changes(current_image, next_image):
    current_image, next_image = np.asarray(current_image), np.asarray(next_image)
    doors = (current_image[..., 0] == OBJECT_TO_IDX["door"]) & (
        next_image[..., 0] == OBJECT_TO_IDX["door"]
    )
    changed = doors & (current_image[..., 2] != next_image[..., 2])
    unlocked = changed & (current_image[..., 2] == STATE_TO_IDX["locked"])
    return int(np.count_nonzero(changed)), int(np.count_nonzero(unlocked))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _explorer_config(memory_mode: str):
    source = OmegaConf.load(ROOT_DIR / "trainer" / "conf" / "config_mac.yaml")
    rmax = source.domains.minigrid.rmax_like
    if int(rmax.rollout_steps) != ROLLOUT_STEPS:
        raise ValueError(
            f"preregistered rollout_steps is {ROLLOUT_STEPS}, config has {rmax.rollout_steps}"
        )
    cfg = OmegaConf.create(
        {
            "domains": {
                "minigrid": {
                    "grid_shape": list(source.domains.minigrid.grid_shape),
                    "obs_norm": list(source.domains.minigrid.obs_norm),
                    "action_norm": int(source.domains.minigrid.action_norm),
                    "rmax_like": OmegaConf.to_container(rmax, resolve=True),
                }
            }
        }
    )
    cfg.domains.minigrid.rmax_like.memory_mode = memory_mode
    return cfg


def evaluate_map(explorer, corpus, map_index: int, reset_seed: int, *, train: bool):
    reachable = reachable_positions(
        corpus.object_maps[map_index], corpus.color_maps[map_index],
        corpus.state_maps[map_index], corpus.inventory_tokens[map_index],
    )
    deadlock = len(reachable) <= 1
    env = make_env(corpus, map_index, max_steps=TRANSITION_BUDGET)
    explorer.set_training(train)
    explorer.begin_rollout(TRANSITION_BUDGET)
    observation, _ = env.reset(seed=reset_seed)
    visited = set()
    movement_count = no_change_count = 0
    pickup_count = drop_count = toggle_count = unlock_count = invalid_count = 0
    interaction_attempts = 0
    coverage_at_256 = None

    for transition_index in range(TRANSITION_BUDGET):
        current_image = _image(observation)
        current_token = carrying_token_from_env(env)
        explorer.set_carrying_token(current_token)
        action = explorer.select_action(current_image)
        environment_mask = np.asarray(
            explorer.environment_action_mask(current_image, current_token).cpu(),
            dtype=bool,
        )
        invalid_count += int(not environment_mask[action])
        next_observation, _, terminated, truncated, _ = env.step(compact_to_native(action))
        next_image = _image(next_observation)
        next_token = carrying_token_from_env(env)
        explorer.set_next_carrying_token(next_token)
        reward = explorer.intrinsic_reward(current_image, action, next_image)
        explorer.record_transition(
            reward,
            bool(terminated or truncated),
            obs_next=next_image,
            action=action,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

        current_position, next_position = _position(current_image), _position(next_image)
        visited.update((current_position, next_position))
        movement_count += int(current_position != next_position)
        no_change_count += int(
            explorer.counter.semantic_state_key(current_image, current_token)
            == explorer.counter.semantic_state_key(next_image, next_token)
        )
        door_changes, unlocks = _door_changes(current_image, next_image)
        pickup = action == 3 and current_token == 0 and next_token != 0
        drop = action == 5 and current_token != 0 and next_token == 0
        toggle = action == 4 and door_changes > 0
        pickup_count += int(pickup)
        drop_count += int(drop)
        toggle_count += int(toggle)
        unlock_count += unlocks
        interaction_attempts += int(action in (3, 4, 5))

        if transition_index + 1 == ROLLOUT_STEPS:
            coverage_at_256 = len(visited.intersection(reachable)) / len(reachable)
        if terminated or truncated:
            # Training record_transition seals and resets the recurrent state
            # itself.  Frozen record_transition intentionally does no writes,
            # so reset explicitly to prevent hidden-state leakage across eval
            # episodes.
            if not train:
                explorer.reset_episode()
            observation, _ = env.reset(seed=reset_seed + transition_index + 1)
        else:
            observation = next_observation

    explorer.mark_rollout_boundary()
    env.close()
    if coverage_at_256 is None:
        raise RuntimeError("coverage@256 was not recorded")
    return {
        "deadlock": deadlock,
        "reachable_positions": len(reachable),
        "unique_reachable_positions": len(visited.intersection(reachable)),
        "coverage_ratio_256": coverage_at_256,
        "coverage_ratio_1000": len(visited.intersection(reachable)) / len(reachable),
        "movement_rate": movement_count / TRANSITION_BUDGET,
        "semantic_no_change_rate": no_change_count / TRANSITION_BUDGET,
        "pickup_successes": pickup_count,
        "drop_successes": drop_count,
        "toggle_successes": toggle_count,
        "unlock_successes": unlock_count,
        "interaction_attempt_rate": interaction_attempts / TRANSITION_BUDGET,
        "invalid_environment_action_rate": invalid_count / TRANSITION_BUDGET,
        "frozen_evaluation": not train,
    }


def _evaluate_variant_seed(corpus, memory_mode, seed):
    rows = []
    variant = "memory_on" if memory_mode == "episodic" else "memory_off"
    _set_seed(seed)
    explorer = MiniGridRMaxExplorer(_explorer_config(memory_mode))
    for start in range(0, len(corpus), JOINT_ITERATION_MAPS):
        map_indices = range(start, min(start + JOINT_ITERATION_MAPS, len(corpus)))
        explorer.begin_iteration()
        for map_index in map_indices:
            evaluate_map(
                explorer, corpus, map_index,
                reset_seed=1_000_000 + seed * 10_000 + map_index * 1_001,
                train=True,
            )
        update = explorer.end_iteration()
        if not update["updated"]:
            raise RuntimeError(f"joint iteration starting at map {start} did not update")

        # Report post-update frozen rollouts.  They exercise the exact policy
        # that will be reused on later maps without appending PPO/RIDE data or
        # changing either map-local novelty table.
        for map_index in map_indices:
            metrics = evaluate_map(
                explorer, corpus, map_index,
                reset_seed=2_000_000 + seed * 10_000 + map_index * 1_001,
                train=False,
            )
            rows.append(
                {
                    "variant": variant,
                    "memory_mode": memory_mode,
                    "seed": seed,
                    "map_index": map_index,
                    "iteration": int(corpus.iterations[map_index]),
                    "batch_index": int(corpus.batch_indices[map_index]),
                    "map_hash": str(corpus.map_hashes[map_index]),
                    "block": map_index // BLOCK_SIZE,
                    "joint_iteration": start // JOINT_ITERATION_MAPS,
                    "training_maps_in_joint_iteration": len(map_indices),
                    "ppo_updates_in_joint_iteration": int(update["updated"]),
                    "ppo_update_count_total": explorer.update_count,
                    **metrics,
                }
            )
        explorer.set_training(True)
    print(f"[Explorer A/B] completed {variant}, seed={seed}")
    return rows


def _evaluate_variant_seed_from_path(task):
    corpus_path, memory_mode, seed = task
    torch.set_num_threads(1)
    corpus = load_corpus(corpus_path, expected_size=EXPECTED_MAPS)
    return _evaluate_variant_seed(corpus, memory_mode, seed)


def run_experiment(corpus, corpus_path=None):
    tasks = [
        (str(corpus_path), memory_mode, seed)
        for memory_mode in VARIANTS
        for seed in EXPLORER_SEEDS
    ]
    if corpus_path is not None and not torch.cuda.is_available():
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(tasks), mp_context=context
        ) as executor:
            grouped_rows = executor.map(_evaluate_variant_seed_from_path, tasks)
            return [row for group in grouped_rows for row in group]

    rows = []
    for _, memory_mode, seed in tasks:
        rows.extend(_evaluate_variant_seed(corpus, memory_mode, seed))
    return rows


def _mean(rows, variant, field, *, seed=None, block=None):
    values = [
        float(row[field]) for row in rows
        if row["variant"] == variant and not row["deadlock"]
        and (seed is None or row["seed"] == seed)
        and (block is None or row["block"] == block)
    ]
    return float(np.mean(values)) if values else float("nan")


def _paired_bootstrap(rows, samples=BOOTSTRAP_SAMPLES):
    pairs = defaultdict(dict)
    for row in rows:
        if not row["deadlock"]:
            pairs[(row["seed"], row["map_index"])][row["variant"]] = float(
                row["coverage_ratio_1000"]
            )
    by_seed = defaultdict(list)
    for (seed, _), values in pairs.items():
        if set(values) != {"memory_on", "memory_off"}:
            raise ValueError("incomplete paired A/B result")
        by_seed[seed].append(values["memory_on"] - values["memory_off"])
    seeds = np.asarray(EXPLORER_SEEDS)
    rng = np.random.default_rng(20260902)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled = []
        for seed in selected_seeds:
            differences = np.asarray(by_seed[int(seed)])
            sampled.extend(rng.choice(differences, size=len(differences), replace=True))
        draws[index] = np.mean(sampled)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(rows):
    overall = {
        variant: {
            field: _mean(rows, variant, field)
            for field in (
                "coverage_ratio_256", "coverage_ratio_1000", "movement_rate",
                "semantic_no_change_rate", "interaction_attempt_rate",
                "invalid_environment_action_rate",
            )
        }
        for variant in ("memory_on", "memory_off")
    }
    per_seed = {}
    for seed in EXPLORER_SEEDS:
        on = _mean(rows, "memory_on", "coverage_ratio_1000", seed=seed)
        off = _mean(rows, "memory_off", "coverage_ratio_1000", seed=seed)
        per_seed[str(seed)] = {"memory_on": on, "memory_off": off, "difference": on - off}
    blocks = []
    for block in range(EXPECTED_MAPS // BLOCK_SIZE):
        block_row = {"block": block, "map_start": block * BLOCK_SIZE, "map_end": (block + 1) * BLOCK_SIZE - 1}
        for variant in ("memory_on", "memory_off"):
            block_row[f"{variant}_coverage_256"] = _mean(rows, variant, "coverage_ratio_256", block=block)
            block_row[f"{variant}_coverage_1000"] = _mean(rows, variant, "coverage_ratio_1000", block=block)
        blocks.append(block_row)
    difference = overall["memory_on"]["coverage_ratio_1000"] - overall["memory_off"]["coverage_ratio_1000"]
    confidence_interval = _paired_bootstrap(rows)
    last_block_256_difference = blocks[-1]["memory_on_coverage_256"] - blocks[-1]["memory_off_coverage_256"]
    cross_map_learning = {}
    for variant in ("memory_on", "memory_off"):
        first = blocks[0][f"{variant}_coverage_256"]
        last = blocks[-1][f"{variant}_coverage_256"]
        cross_map_learning[variant] = {
            "first_80_coverage_256": first,
            "last_80_coverage_256": last,
            "last_minus_first": last - first,
        }
    seed_wins = sum(value["difference"] > 0.0 for value in per_seed.values())
    acceptance = {
        "overall_gain_at_least_5pp": difference >= 0.05,
        "ci_lower_bound_above_zero": confidence_interval[0] > 0.0,
        "at_least_two_seed_wins": seed_wins >= 2,
        "last_80_zero_shot_gain": last_block_256_difference > 0.0,
    }
    acceptance["passed"] = all(acceptance.values())
    return {
        "protocol": {
            "maps": EXPECTED_MAPS, "seeds": list(EXPLORER_SEEDS),
            "transition_budget": TRANSITION_BUDGET, "rollout_steps": ROLLOUT_STEPS,
            "joint_iteration_maps": JOINT_ITERATION_MAPS,
            "reporting": "post_joint_update_frozen_rollouts",
            "ppo_update_semantics": "one joint update per up-to-eight training maps",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "nondeadlock_rows_per_variant": sum(
            row["variant"] == "memory_on" and not row["deadlock"] for row in rows
        ),
        "deadlock_maps": len({row["map_index"] for row in rows if row["deadlock"]}),
        "overall": overall,
        "coverage_1000_difference": difference,
        "coverage_1000_difference_95ci": confidence_interval,
        "last_80_coverage_256_difference": last_block_256_difference,
        "cross_map_learning": cross_map_learning,
        "per_seed": per_seed,
        "blocks": blocks,
        "acceptance": acceptance,
    }


def save_results(rows, summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_map.csv"
    summary_path = output_dir / "summary.json"
    figure_path = output_dir / "coverage_by_block.png"
    existing = [path for path in (csv_path, summary_path, figure_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite A/B results: {existing}")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    x = np.arange(len(summary["blocks"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, suffix, title in zip(axes, ("256", "1000"), ("coverage@256", "coverage@1000")):
        for variant in ("memory_on", "memory_off"):
            axis.plot(
                x, [row[f"{variant}_coverage_{suffix}"] for row in summary["blocks"]],
                marker="o", label=variant,
            )
        axis.set_title(title)
        axis.set_xlabel("80-map block")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("reachable-cell coverage ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return csv_path, summary_path, figure_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare episodic memory against a capacity-matched no-memory explorer."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "per_map.csv",
        output_dir / "summary.json",
        output_dir / "coverage_by_block.png",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite A/B results: {existing}")
    corpus = load_corpus(args.corpus, expected_size=EXPECTED_MAPS)
    rows = run_experiment(corpus, corpus_path=args.corpus.expanduser().resolve())
    summary = summarize(rows)
    summary["protocol"]["corpus_hash"] = corpus.metadata["corpus_hash"]
    rmax_config = OmegaConf.to_container(
        _explorer_config("episodic").domains.minigrid.rmax_like, resolve=True
    )
    rmax_config["memory_mode"] = list(VARIANTS)
    summary["protocol"]["rmax_config"] = rmax_config
    paths = save_results(rows, summary, output_dir)
    print(json.dumps(summary["acceptance"], indent=2, sort_keys=True))
    print(f"[Explorer A/B] results: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
