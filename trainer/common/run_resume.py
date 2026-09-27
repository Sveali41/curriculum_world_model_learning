"""Iteration-boundary state for future Crafter DR/MAC runs."""

import csv
import hashlib
import json
import os
import random
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def config_digest(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("resume_training", None)
    config.pop("save_iteration_state", None)
    config.pop("save_wm_update_checkpoints", None)
    config.pop("force_fresh_start", None)
    config["generator_agent"].pop("total_iterations", None)
    # MAC assigns this to each temporary iteration archive at runtime.
    config["attention_model"].pop("data_dir", None)
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume state requires CUDA, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def replay_state(buffer):
    return {"buffer": buffer.buffer, "rng": buffer._rng.bit_generator.state}


def restore_replay(buffer, state):
    if len(state["buffer"]) > buffer.max_size:
        raise ValueError("Replay state exceeds configured fisher_buffer_size")
    buffer.buffer = state["buffer"]
    buffer._rng.bit_generator.state = state["rng"]


def generator_state(interface, include_policy):
    state = {
        "prev_data": interface.prev_data,
        "elite_buffer": interface.elite_buffer,
        "stage_rng": interface._crafter_stage_rng.bit_generator.state,
        "lp_scale_history": {key: list(value) for key, value in interface._crafter_lp_scale_history.items()},
        "diversity_archive": interface.diversity.archive,
        "diversity_model": interface.diversity.state_dict(),
    }
    if include_policy:
        ppo = interface.ppo
        state["ppo"] = {
            "policy": ppo.policy.state_dict(),
            "policy_old": ppo.policy_old.state_dict(),
            "encoder": ppo.encoder.state_dict() if ppo.encoder is not None else None,
            "optimizer": ppo.optimizer.state_dict(),
            "buffer": ppo.buffer,
            "round_lengths": list(ppo.round_lengths),
            "current_round_count": ppo.current_round_count,
            "last_mean_reward": ppo.last_mean_reward,
            "last_entropy_coef": ppo.last_entropy_coef,
            "shared_actor_gradient_conflict_updates": ppo._shared_actor_gradient_conflict_updates,
            "shared_actor_gradient_measure_count": ppo._shared_actor_gradient_measure_count,
        }
    return state


def restore_generator(interface, state, include_policy):
    interface.prev_data = state["prev_data"]
    if interface.prev_data is not None:
        interface.prev_data = tuple(
            value.to(interface.device) if torch.is_tensor(value) else value
            for value in interface.prev_data
        )
    interface.elite_buffer = state["elite_buffer"]
    interface._crafter_stage_rng.bit_generator.state = state["stage_rng"]
    for key, values in state["lp_scale_history"].items():
        interface._crafter_lp_scale_history[key] = deque(values, maxlen=5)
    interface.diversity.load_state_dict(state["diversity_model"])
    interface.diversity.archive = state["diversity_archive"]
    if include_policy:
        ppo = interface.ppo
        saved = state["ppo"]
        ppo.policy.load_state_dict(saved["policy"])
        ppo.policy_old.load_state_dict(saved["policy_old"])
        if (ppo.encoder is None) != (saved["encoder"] is None):
            raise ValueError("Generator encoder configuration differs from resume state")
        if ppo.encoder is not None:
            ppo.encoder.load_state_dict(saved["encoder"])
        ppo.optimizer.load_state_dict(saved["optimizer"])
        ppo.buffer = saved["buffer"]
        ppo.round_lengths = deque(saved["round_lengths"])
        ppo.current_round_count = saved["current_round_count"]
        ppo.last_mean_reward = saved["last_mean_reward"]
        ppo.last_entropy_coef = saved["last_entropy_coef"]
        ppo._shared_actor_gradient_conflict_updates = saved["shared_actor_gradient_conflict_updates"]
        ppo._shared_actor_gradient_measure_count = saved["shared_actor_gradient_measure_count"]


def guard_fresh_results(csv_path, state_path, seed):
    """Prevent a fresh run from silently mixing with an older same-seed run."""
    if Path(state_path).exists():
        raise FileExistsError(
            f"Iteration state already exists: {state_path}. Choose a new result directory "
            "or use resume_training=true with force_fresh_start=false."
        )
    path = Path(csv_path)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Seed" not in reader.fieldnames:
            raise ValueError(f"Existing result CSV has no Seed column: {path}")
        try:
            existing = any(int(row["Seed"]) == int(seed) for row in reader)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Existing result CSV has invalid Seed values: {path}") from exc
    if existing:
        raise FileExistsError(
            f"Result CSV already has seed {seed}: {path}. Choose a new result directory "
            "or use resume_training=true with a matching iteration state."
        )


def validate_result_rows(csv_path, seed, completed_iteration, expected_columns=None):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume requires result CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Seed" not in reader.fieldnames or "Iter" not in reader.fieldnames:
            raise ValueError(f"Resume CSV has no Seed/Iter columns: {path}")
        if expected_columns is not None and reader.fieldnames != list(expected_columns):
            raise ValueError(f"Resume CSV schema differs from current configuration: {path}")
        try:
            iterations = [int(row["Iter"]) for row in reader if int(row["Seed"]) == int(seed)]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Resume CSV contains an invalid Seed/Iter: {path}") from exc
    expected = list(range(1, int(completed_iteration) + 1))
    if iterations != expected:
        raise ValueError(
            f"Resume CSV for seed {seed} has iterations {iterations}; state requires {expected}. "
            "Use the matching state and CSV without extra or missing rows. "
            "A crash inside the next iteration may also leave partial sidecar rows; "
            "move those rows aside before retrying."
        )


def validate_sidecars(csv_paths, jsonl_paths, seed, completed_iteration):
    """Reject partial next-iteration diagnostics before appending to them."""
    for path in map(Path, csv_paths):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            seed_key = "Seed" if "Seed" in fields else "seed"
            iter_key = "Iter" if "Iter" in fields else "iteration"
            if seed_key not in fields or iter_key not in fields:
                raise ValueError(f"Resume sidecar lacks seed/iteration columns: {path}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    row_seed = int(row[seed_key])
                    row_iteration = int(row[iter_key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid resume sidecar row at {path}:{line_number}") from exc
                if row_seed == int(seed) and row_iteration > int(completed_iteration):
                    raise ValueError(
                        f"Resume sidecar {path}:{line_number} has iteration {row_iteration} "
                        f"past saved iteration {completed_iteration}. Move partial next-iteration "
                        "rows aside before resuming; no files were changed."
                    )
    for path in map(Path, jsonl_paths):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    row_seed = int(row["Seed"])
                    row_iteration = int(row["Iter"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid resume map sidecar row at {path}:{line_number}") from exc
                if row_seed == int(seed) and row_iteration > int(completed_iteration):
                    raise ValueError(
                        f"Resume map sidecar {path}:{line_number} has iteration {row_iteration} "
                        f"past saved iteration {completed_iteration}. Move partial next-iteration "
                        "rows aside before resuming; no files were changed."
                    )


def save_state(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path, cfg, kind, csv_path, expected_columns=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume state missing: {path}. An ordinary WM checkpoint is insufficient.")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 1 or state.get("kind") != kind:
        raise ValueError(f"Incompatible {kind} resume state: {path}")
    if state.get("config_digest") != config_digest(cfg):
        raise ValueError("Resume config differs from saved run; only total_iterations may increase")
    completed = int(state["completed_iteration"])
    if int(cfg.generator_agent.total_iterations) < completed:
        raise ValueError("total_iterations is less than the completed resume iteration")
    validate_result_rows(csv_path, int(cfg.seed), completed, expected_columns)
    return state
