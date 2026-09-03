"""Evaluate cross-map accumulation of the MiniGrid intrinsic explorer.

The evaluator deliberately keeps the corpus and held-out split local to this
file.  It is therefore safe to use it with a frozen corpus without changing
the training or data-collection protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
WM_ROOT = ROOT_DIR / "wm"
sys.path.insert(0, str(WM_ROOT))
sys.path.insert(1, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from minigrid.core.constants import IDX_TO_COLOR, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.wrappers import FullyObsWrapper
from omegaconf import OmegaConf

from domain.minigrid.minigrid_custom_env import CustomMiniGridEnv
from domain.minigrid.action_codec import carrying_token_from_env, compact_to_native
from modelBased.exploration.count_based import MINIGRID_PLAYER_ID
from modelBased.exploration.minigrid_corpus import (
    load_corpus,
    map_strings,
    reachable_positions,
)
from modelBased.data.data_collect import collect_minigrid_interaction_transitions
from modelBased.exploration.minigrid_rmax import MiniGridRMaxExplorer


SEEDS = (17, 29, 41)
VARIANTS = ("persistent", "reset_per_map", "random")
EXPECTED_MAPS = 400
BLOCK_SIZE = 80
HELDOUT_PER_BLOCK = 16
TRAIN_PER_BLOCK = BLOCK_SIZE - HELDOUT_PER_BLOCK
TRAIN_TRANSITION_BUDGET = 1000
EVAL_TRANSITION_BUDGET = 256
MILESTONES = (0, 64, 128, 192, 256, 320)
BOOTSTRAP_SAMPLES = 10_000
ACTION_COUNT = 6


def _image(observation):
    return observation["image"] if isinstance(observation, dict) else observation


def _position(image):
    positions = np.argwhere(np.asarray(image)[..., 0] == MINIGRID_PLAYER_ID)
    if len(positions) != 1:
        raise ValueError(f"expected exactly one observed agent, found {len(positions)}")
    # FullyObsWrapper emits [x, y, channel], while corpus reachability uses
    # the layout's [row/y, column/x] coordinates.
    x, y = (int(value) for value in positions[0])
    return y, x


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _explorer_config(corpus):
    source = OmegaConf.load(ROOT_DIR / "trainer" / "conf" / "config_mac.yaml")
    # Keep the optional checkpoint interpolation unresolved.  The evaluator
    # never writes checkpoints and should also work when MODEL_FPATH is not
    # present in a lightweight test environment.
    rmax = OmegaConf.to_container(source.domains.minigrid.rmax_like, resolve=False)
    rmax["rollout_steps"] = EVAL_TRANSITION_BUDGET
    rmax["checkpoint_path"] = None
    return OmegaConf.create(
        {
            "seed": int(getattr(source, "seed", 0)),
            "domains": {
                "minigrid": {
                    "grid_shape": [
                        3,
                        int(corpus.object_maps.shape[1]),
                        int(corpus.object_maps.shape[2]),
                    ],
                    "obs_norm": list(source.domains.minigrid.obs_norm),
                    "action_norm": ACTION_COUNT,
                    "rmax_like": rmax,
                }
            },
        }
    )


def _make_env(corpus, map_index: int, start_dir: int, max_steps: int):
    layout, colors = map_strings(
        corpus.object_maps[map_index],
        corpus.color_maps[map_index],
        corpus.state_maps[map_index],
    )
    token = int(corpus.inventory_tokens[map_index])
    initial_color = None if token == 0 else IDX_TO_COLOR[token - 1]
    return FullyObsWrapper(
        CustomMiniGridEnv(
            layout_str=layout,
            color_str=colors,
            agent_start_dir=int(start_dir),
            initial_carrying_key_color=initial_color,
            max_steps=int(max_steps),
        )
    )


def _door_changes(current_image, next_image):
    current_image, next_image = np.asarray(current_image), np.asarray(next_image)
    doors = (current_image[..., 0] == OBJECT_TO_IDX["door"]) & (
        next_image[..., 0] == OBJECT_TO_IDX["door"]
    )
    changed = doors & (current_image[..., 2] != next_image[..., 2])
    unlocked = changed & (current_image[..., 2] == STATE_TO_IDX["locked"])
    return int(np.count_nonzero(changed)), int(np.count_nonzero(unlocked))


def _semantic_key(explorer, image, token):
    return explorer.counter.semantic_state_key(image, int(token))


def _empty_metrics():
    return {
        "reachable_positions": 0,
        "unique_reachable_positions": 0,
        "coverage_ratio_256": float("nan"),
        "coverage_ratio_1000": float("nan"),
        "absolute_unique_positions_256": 0,
        "absolute_unique_positions_1000": 0,
        "movement_rate": 0.0,
        "semantic_no_change_rate": 0.0,
        "pickup_successes": 0,
        "drop_successes": 0,
        "toggle_successes": 0,
        "unlock_successes": 0,
        "interaction_attempt_rate": 0.0,
        "invalid_environment_action_rate": 0.0,
        "invalid_environment_actions": 0,
        "interaction_top1_accuracy": float("nan"),
    }


def rollout_map(
    explorer, corpus, map_index, start_dir, *, seed, train, random_actions=False
):
    """Run a map rollout and optionally append transitions to PPO."""
    budget = TRAIN_TRANSITION_BUDGET if train else EVAL_TRANSITION_BUDGET
    reachable = reachable_positions(
        corpus.object_maps[map_index],
        corpus.color_maps[map_index],
        corpus.state_maps[map_index],
        corpus.inventory_tokens[map_index],
    )
    env = _make_env(corpus, map_index, start_dir, budget)
    explorer.set_training(bool(train))
    explorer.begin_rollout(budget)
    observation, _ = env.reset(seed=int(seed))
    visited = set()
    metrics = _empty_metrics()
    interaction_attempts = 0
    semantic_no_change = 0
    movement = 0
    invalid_actions = 0
    coverage_at_256 = None
    initial_token = int(corpus.inventory_tokens[map_index])
    action_rng = np.random.default_rng(int(seed))

    for transition_index in range(budget):
        current_image = _image(observation)
        current_token = carrying_token_from_env(env)
        explorer.set_carrying_token(current_token)
        environment_mask = np.asarray(
            explorer.environment_action_mask(current_image, current_token).detach().cpu(),
            dtype=bool,
        )
        if random_actions:
            action = int(action_rng.integers(ACTION_COUNT))
        else:
            action = int(explorer.select_action(current_image))
        invalid_actions += int(not environment_mask[action])
        next_observation, _, terminated, truncated, _ = env.step(
            compact_to_native(action)
        )
        next_image = _image(next_observation)
        next_token = carrying_token_from_env(env)
        explorer.set_next_carrying_token(next_token)
        current_semantic = _semantic_key(explorer, current_image, current_token)
        next_semantic = _semantic_key(explorer, next_image, next_token)
        reward = explorer.intrinsic_reward(current_image, action, next_image)
        if train:
            explorer.record_transition(
                reward,
                bool(terminated or truncated),
                terminated=bool(terminated),
                truncated=bool(truncated),
                obs_next=next_image,
                action=action,
            )

        current_position, next_position = _position(current_image), _position(
            next_image
        )
        visited.update((current_position, next_position))
        movement += int(current_position != next_position)
        semantic_no_change += int(current_semantic == next_semantic)
        door_changes, unlocks = _door_changes(current_image, next_image)
        pickup = action == 3 and current_token == 0 and next_token != 0
        drop = action == 5 and current_token != 0 and next_token == 0
        toggle = action == 4 and door_changes > 0
        metrics["pickup_successes"] += int(pickup)
        metrics["drop_successes"] += int(drop)
        metrics["toggle_successes"] += int(toggle)
        metrics["unlock_successes"] += unlocks
        is_interaction = action in (3, 4, 5)
        interaction_attempts += int(is_interaction)

        if transition_index + 1 == EVAL_TRANSITION_BUDGET:
            coverage_at_256 = len(visited.intersection(reachable)) / max(
                len(reachable), 1
            )
        if terminated or truncated:
            if not train:
                explorer.reset_episode()
            observation, _ = env.reset(seed=int(seed) + transition_index + 1)
        else:
            observation = next_observation

    explorer.mark_rollout_boundary()
    env.close()
    if coverage_at_256 is None:
        raise RuntimeError("coverage@256 was not recorded")
    metrics.update(
        {
            "reachable_positions": len(reachable),
            "unique_reachable_positions": len(visited.intersection(reachable)),
            "coverage_ratio_256": coverage_at_256,
            "coverage_ratio_1000": len(visited.intersection(reachable))
            / max(len(reachable), 1),
            "absolute_unique_positions_256": len(visited),
            "absolute_unique_positions_1000": len(visited),
            "movement_rate": movement / budget,
            "semantic_no_change_rate": semantic_no_change / budget,
            "interaction_attempt_rate": interaction_attempts / budget,
            "invalid_environment_action_rate": invalid_actions / budget,
            "invalid_environment_actions": invalid_actions,
            "interaction_top1_accuracy": float("nan"),
            "initial_inventory_token": initial_token,
        }
    )
    return metrics


def stratified_split(corpus):
    """Split each 80-map block into 64 train and 16 held-out maps."""
    if len(corpus) != EXPECTED_MAPS:
        raise ValueError(f"expected exactly {EXPECTED_MAPS} maps, found {len(corpus)}")
    hashes = [str(value) for value in corpus.map_hashes]
    if len(set(hashes)) != len(hashes):
        raise ValueError("the frozen corpus contains duplicate map hashes")
    train, heldout = [], []
    for block_start in range(0, EXPECTED_MAPS, BLOCK_SIZE):
        block = list(range(block_start, block_start + BLOCK_SIZE))
        ordered = sorted(block, key=lambda i: (str(corpus.map_hashes[i]), i))
        # Taking every fifth hash spreads held-out examples throughout each
        # deterministic block while preserving equal train/held-out strata.
        block_heldout = set(ordered[::5][:HELDOUT_PER_BLOCK])
        heldout.extend(sorted(block_heldout))
        train.extend(i for i in block if i not in block_heldout)
    train_hashes = {str(corpus.map_hashes[i]) for i in train}
    heldout_hashes = {str(corpus.map_hashes[i]) for i in heldout}
    if train_hashes & heldout_hashes:
        raise ValueError("train and held-out corpus hashes overlap")
    if len(train) != 320 or len(heldout) != 80:
        raise AssertionError("stratified split did not produce 320/80 maps")
    return train, heldout


def _new_explorer(corpus, seed):
    cfg = _explorer_config(corpus)
    cfg.seed = int(seed)
    # Explorer owns a torch Generator for sampling, while module
    # initialization uses the process-global RNG.  Isolate both so equal seed
    # variants have byte-identical initial weights and do not perturb callers.
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        return MiniGridRMaxExplorer(cfg)


def _clone_for_evaluation(source, corpus, seed):
    clone = _new_explorer(corpus, seed)
    clone.ppo.policy.load_state_dict(source.ppo.policy.state_dict())
    clone.ppo.policy_old.load_state_dict(source.ppo.policy_old.state_dict())
    clone.ride.representation.load_state_dict(source.ride.representation.state_dict())
    clone.ppo.policy.eval()
    clone.ppo.policy_old.eval()
    clone.ride.representation.eval()
    clone.set_training(False)
    return clone


def _staged_interaction_top1(explorer, corpus, map_index, seed):
    """Measure argmax action accuracy on real staged key/door transitions."""
    env = _make_env(
        corpus, map_index, int(corpus.start_dirs[map_index]), EVAL_TRANSITION_BUDGET
    )
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "env": {
                "env_path": f"corpus-map-{map_index}-{corpus.map_hashes[map_index]}"
            },
        }
    )
    try:
        data = collect_minigrid_interaction_transitions(env, cfg, sample_count=24)
    except ValueError:
        return float("nan"), 0
    finally:
        # The collector stages and resets this environment internally; no
        # state from it is used by the normal held-out rollout.
        env.close()

    observations, _, actions, _, _, infos, labels = data
    expected_types = {"key_pickup", "key_drop", "normal_door", "locked_success"}
    correct = total = 0
    explorer.begin_rollout(1)
    for index, label in enumerate(np.asarray(labels).astype(str).tolist()):
        if label not in expected_types:
            continue
        info = infos[index]
        token = int(info["current_carrying_token"])
        observation = np.asarray(observations[index])
        explorer.set_carrying_token(token)
        explorer.reset_episode()
        state = explorer._policy_state(observation, token)
        mask = torch.ones(ACTION_COUNT, dtype=torch.bool)
        with torch.no_grad():
            distribution, _, _ = explorer.ppo.policy_old.step(
                state, explorer.ppo.initial_hidden(), mask
            )
        predicted = int(torch.argmax(distribution.logits, dim=-1).item())
        expected = int(actions[index])
        total += 1
        correct += int(predicted == expected)
    explorer.reset_episode()
    return (correct / total if total else float("nan")), total


def _evaluate_variant(corpus, train_indices, heldout_indices, seed, variant):
    rows = []
    explorer = _new_explorer(corpus, seed)
    trained = 0
    for milestone in MILESTONES:
        if variant == "persistent":
            while trained < milestone:
                explorer.begin_iteration()
                for map_index in train_indices[trained : trained + 8]:
                    rollout_map(
                        explorer,
                        corpus,
                        map_index,
                        int(corpus.start_dirs[map_index]),
                        seed=1_000_000 + seed * 10_000 + map_index * 1_001,
                        train=True,
                    )
                explorer.end_iteration()
                trained += 8
        elif variant == "reset_per_map":
            # Train the same architecture for the same per-map transition
            # budget, but discard it before the next training map.  The model
            # evaluated at a milestone therefore contains knowledge from only
            # the most recent map, providing a trained non-accumulating
            # baseline rather than an untrained-policy comparison.
            while trained < milestone:
                for map_index in train_indices[trained : trained + 8]:
                    explorer = _new_explorer(corpus, seed)
                    explorer.begin_iteration()
                    rollout_map(
                        explorer,
                        corpus,
                        map_index,
                        int(corpus.start_dirs[map_index]),
                        seed=1_000_000 + seed * 10_000 + map_index * 1_001,
                        train=True,
                    )
                    explorer.end_iteration()
                trained += 8
        for map_index in heldout_indices:
            staged_accuracy, staged_count = (
                (float("nan"), 0)
                if variant == "random"
                else _staged_interaction_top1(
                    _clone_for_evaluation(explorer, corpus, seed + map_index),
                    corpus,
                    map_index,
                    seed + map_index,
                )
            )
            for start_dir in range(4):
                evaluation_explorer = _clone_for_evaluation(
                    explorer,
                    corpus,
                    seed + map_index * 10 + start_dir,
                )
                metrics = rollout_map(
                    evaluation_explorer,
                    corpus,
                    map_index,
                    start_dir,
                    seed=2_000_000 + seed * 10_000 + map_index * 1_001 + start_dir,
                    train=False,
                    random_actions=variant == "random",
                )
                metrics["interaction_top1_accuracy"] = staged_accuracy
                metrics["interaction_top1_samples"] = staged_count
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "train_maps_seen": milestone,
                        "map_index": map_index,
                        "start_dir": start_dir,
                        "block": map_index // BLOCK_SIZE,
                        "map_hash": str(corpus.map_hashes[map_index]),
                        **metrics,
                    }
                )
    return rows


def _mean(rows, variant, milestone, field, seed=None):
    values = [
        float(row[field])
        for row in rows
        if row["variant"] == variant
        and row["train_maps_seen"] == milestone
        and (seed is None or row["seed"] == seed)
        and np.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else float("nan")


def paired_bootstrap(rows, field="coverage_ratio_256", samples=BOOTSTRAP_SAMPLES):
    clusters = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["train_maps_seen"] == MILESTONES[-1]:
            clusters[(row["seed"], row["map_index"])][row["variant"]].append(
                float(row[field])
            )
    differences = defaultdict(list)
    for key, values in clusters.items():
        if {"persistent", "reset_per_map"} <= set(values):
            differences[key[0]].append(
                float(np.mean(values["persistent"]))
                - float(np.mean(values["reset_per_map"]))
            )
    rng = np.random.default_rng(20260902)
    draws = []
    for _ in range(int(samples)):
        sampled = []
        for seed in SEEDS:
            values = differences.get(seed, [])
            if values:
                sampled.extend(rng.choice(values, size=len(values), replace=True))
        if sampled:
            draws.append(float(np.mean(sampled)))
    return (
        [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        if draws
        else [float("nan"), float("nan")]
    )


def summarize(rows, corpus, train_indices, heldout_indices):
    final_difference = _mean(rows, "persistent", 320, "coverage_ratio_256") - _mean(
        rows, "reset_per_map", 320, "coverage_ratio_256"
    )
    ci = paired_bootstrap(rows)
    absolute_ci = paired_bootstrap(rows, "absolute_unique_positions_256")
    per_seed = {}
    for seed in SEEDS:
        difference = _mean(rows, "persistent", 320, "coverage_ratio_256", seed) - _mean(
            rows, "reset_per_map", 320, "coverage_ratio_256", seed
        )
        per_seed[str(seed)] = {
            "difference": difference,
            "persistent": _mean(rows, "persistent", 320, "coverage_ratio_256", seed),
            "reset_per_map": _mean(
                rows, "reset_per_map", 320, "coverage_ratio_256", seed
            ),
        }
    seed_wins = sum(item["difference"] > 0 for item in per_seed.values())
    persistent_initial = _mean(rows, "persistent", 0, "coverage_ratio_256")
    persistent_final = _mean(rows, "persistent", 320, "coverage_ratio_256")
    interaction_initial = _mean(rows, "persistent", 0, "interaction_top1_accuracy")
    interaction_final = _mean(rows, "persistent", 320, "interaction_top1_accuracy")
    persistent_invalid_rate = _mean(
        rows, "persistent", 320, "invalid_environment_action_rate"
    )
    random_invalid_rate = _mean(rows, "random", 320, "invalid_environment_action_rate")
    acceptance = {
        "overall_gain_at_least_5pp": final_difference >= 0.05,
        "ci_lower_bound_above_zero": ci[0] > 0.0,
        "at_least_two_seed_wins": seed_wins >= 2,
        "absolute_positions_ci_lower_bound_above_zero": absolute_ci[0] > 0.0,
        "interaction_top1_at_least_70pct": interaction_final >= 0.70,
        "interaction_top1_gain_at_least_10pp": (
            interaction_final - interaction_initial >= 0.10
        ),
        "heldout_performance_not_lower": persistent_final >= persistent_initial,
    }
    acceptance["passed"] = all(acceptance.values())
    milestones = []
    for milestone in MILESTONES:
        milestones.append(
            {
                "train_maps_seen": milestone,
                **{
                    f"{variant}_{field}": _mean(rows, variant, milestone, field)
                    for variant in VARIANTS
                    for field in (
                        "coverage_ratio_256",
                        "absolute_unique_positions_256",
                        "movement_rate",
                        "semantic_no_change_rate",
                        "interaction_top1_accuracy",
                    )
                },
            }
        )
    return {
        "protocol": {
            "maps": len(corpus),
            "train_maps": len(train_indices),
            "heldout_maps": len(heldout_indices),
            "seeds": list(SEEDS),
            "maps_per_iteration": 8,
            "reset_per_map_baseline": (
                "fresh seeded initialization, one-map training, then discard"
            ),
            "train_transition_budget": TRAIN_TRANSITION_BUDGET,
            "eval_transition_budget": EVAL_TRANSITION_BUDGET,
            "milestones": list(MILESTONES),
            "start_directions": 4,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "action_mask_mode": "none_v1",
            "random_baseline": "uniform_compact6_v1",
            "corpus_hash": corpus.metadata.get("corpus_hash"),
            "train_hashes_sha256": hashlib.sha256(
                "".join(
                    sorted(str(corpus.map_hashes[i]) for i in train_indices)
                ).encode()
            ).hexdigest(),
            "heldout_hashes_sha256": hashlib.sha256(
                "".join(
                    sorted(str(corpus.map_hashes[i]) for i in heldout_indices)
                ).encode()
            ).hexdigest(),
        },
        "overall": {
            variant: {
                field: _mean(rows, variant, 320, field)
                for field in (
                    "coverage_ratio_256",
                    "absolute_unique_positions_256",
                    "movement_rate",
                    "semantic_no_change_rate",
                    "pickup_successes",
                    "drop_successes",
                    "toggle_successes",
                    "unlock_successes",
                    "invalid_environment_actions",
                    "invalid_environment_action_rate",
                    "interaction_top1_accuracy",
                )
            }
            for variant in VARIANTS
        },
        "coverage_256_difference": final_difference,
        "coverage_256_difference_95ci": ci,
        "absolute_unique_positions_256_difference_95ci": absolute_ci,
        "invalid_environment_action_rates": {
            "persistent": persistent_invalid_rate,
            "random": random_invalid_rate,
            "persistent_minus_random": persistent_invalid_rate - random_invalid_rate,
        },
        "per_seed": per_seed,
        "milestones": milestones,
        "acceptance": acceptance,
    }


def save_results(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "per_map.csv",
        output_dir / "summary.json",
        output_dir / "coverage_by_milestone.png",
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite transfer results: {existing}")
    with paths[0].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths[1].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    x = np.asarray(MILESTONES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, field, title in zip(
        axes,
        ("coverage_ratio_256", "absolute_unique_positions_256"),
        ("coverage@256", "absolute unique positions@256"),
    ):
        for variant in VARIANTS:
            axis.plot(
                x,
                [row[f"{variant}_{field}"] for row in summary["milestones"]],
                marker="o",
                label=variant,
            )
        axis.set_title(title)
        axis.set_xlabel("training maps")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180)
    plt.close(fig)
    return paths


def run_experiment(corpus):
    train_indices, heldout_indices = stratified_split(corpus)
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            _set_seed(seed)
            rows.extend(
                _evaluate_variant(corpus, train_indices, heldout_indices, seed, variant)
            )
    return rows, train_indices, heldout_indices


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniGrid explorer cross-map transfer."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus, expected_size=EXPECTED_MAPS)
    rows, train_indices, heldout_indices = run_experiment(corpus)
    summary = summarize(rows, corpus, train_indices, heldout_indices)
    paths = save_results(rows, summary, args.output_dir.expanduser().resolve())
    print(json.dumps(summary["acceptance"], indent=2, sort_keys=True))
    print(f"[Explorer transfer] results: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
