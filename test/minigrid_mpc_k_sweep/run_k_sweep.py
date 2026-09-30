#!/usr/bin/env python3
"""Run paired MiniGrid MPC evaluations over WM checkpoints, targets, and H=K."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = ROOT / "wm"
EXPERIMENT_ROOT = ROOT / "test" / "minigrid_mpc_k_sweep"
TARGET_ROOT = ROOT / "trainer" / "level" / "minigrid" / "target_task"

PLANNER_FIELDS = (
    "episode", "seed", "success", "environment_steps", "real_reward",
    "realized_goal_guide_return", "real_reward_plus_realized_goal_guide",
    "dense_return", "reward_mode", "mean_plan_score",
    "mean_plan_native_return", "mean_plan_goal_guide_score",
    "mean_plan_failure_penalty", "mean_plan_legacy_dense_return", "plan_calls",
    "early_replans", "mean_executed_steps_per_plan", "imagined_transitions",
    "mean_planning_latency_ms", "wm_real_pose_match_rate",
    "wm_real_inventory_match_rate", "wm_checkpoint",
)
META_FIELDS = (
    "baseline", "model_id", "checkpoint", "checkpoint_sha256", "target_id",
    "target", "horizon", "execute_steps", "eval_seed",
)
RESULT_FIELDS = META_FIELDS + PLANNER_FIELDS


def _csv_ints(value: str, name: str, minimum: int = 1) -> list[int]:
    try:
        values = list(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Unsafe empty/path component: {value!r}")
    return safe


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Model manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"baseline", "model_id", "checkpoint"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest needs columns {sorted(required)}: {path}"
            )
        models = []
        seen = set()
        for line, raw in enumerate(reader, start=2):
            baseline = (raw.get("baseline") or "").strip()
            model_id = (raw.get("model_id") or "").strip()
            checkpoint_text = os.path.expandvars(os.path.expanduser(
                (raw.get("checkpoint") or "").strip()
            ))
            if not baseline or not model_id or not checkpoint_text:
                raise ValueError(f"Manifest row {line} has an empty required field")
            key = (baseline, model_id)
            if key in seen:
                raise ValueError(f"Duplicate baseline/model_id at row {line}: {key}")
            seen.add(key)
            checkpoint = Path(checkpoint_text)
            if not checkpoint.is_absolute():
                checkpoint = (ROOT / checkpoint).resolve()
            else:
                checkpoint = checkpoint.resolve()
            models.append({
                "baseline": baseline,
                "model_id": model_id,
                "checkpoint": str(checkpoint),
            })
    if not models:
        raise ValueError(f"Model manifest has no rows: {path}")
    return models


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(ROOT),
        "TRAINER_ROOT": str(ROOT),
        "WM_ROOT": str(WM_ROOT),
        "TRAINER_PATH": str(ROOT / "trainer"),
        "ENV_PATH": str(ROOT / "trainer" / "level"),
        "WORLD_MODEL_PATH": str(WM_ROOT / "modelBased"),
        "MODEL_FPATH": str(WM_ROOT / "modelBased" / "models"),
        "TRAIN_DATASET_PATH": str(WM_ROOT / "modelBased" / "data" / "train_world_model"),
    })
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(WM_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _command(
    *, model: dict[str, str], target_id: int, horizon: int, eval_seed: int,
    episodes: int, max_ep_len: int, reward_mode: str, guide_weight: float,
    population: int, elite_count: int, iterations: int, gamma: float,
    cpu_threads: int, case_root: Path,
) -> list[str]:
    target = f"target_task{target_id}"
    layout = TARGET_ROOT / f"{target}.txt"
    planner_output = case_root / "planner"
    hydra_output = case_root / f"hydra_{time.strftime('%Y%m%d_%H%M%S')}"
    return [
        sys.executable, "-u", "-m", "modelBased.policy_training.planners.mpc_planner",
        "domain=minigrid",
        f"domains.minigrid.task_name={target}",
        f"PPO.env_path={layout}",
        "PPO.wm_control_mode=mpc",
        f"PPO.seed={eval_seed}",
        f"PPO.checkpoint_path_wm={model['checkpoint']}",
        f"PPO.max_ep_len={max_ep_len}",
        f"PPO.mpc.episodes={episodes}",
        "PPO.mpc.minigrid_stop_on_prediction_mismatch=false",
        f"PPO.mpc.horizon={horizon}",
        f"PPO.mpc.minigrid_execute_steps={horizon}",
        f"PPO.mpc.minigrid_reward_mode={reward_mode}",
        f"PPO.mpc.minigrid_goal_guide_weight={guide_weight}",
        f"PPO.mpc.population={population}",
        f"PPO.mpc.elite_count={elite_count}",
        f"PPO.mpc.iterations={iterations}",
        f"PPO.mpc.gamma={gamma}",
        f"PPO.mpc.cpu_threads={cpu_threads}",
        f"PPO.mpc.output_dir={planner_output}",
        f"hydra.run.dir={hydra_output}",
    ]


def _read_results(path: Path, *, episodes: int, eval_seed: int, checkpoint: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Planner did not produce episode results: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} episodes in {path}, found {len(rows)}")
    for number, row in enumerate(rows):
        if (int(row["episode"]) != number or int(row["seed"]) != eval_seed + number
                or Path(row["wm_checkpoint"]).resolve() != checkpoint):
            raise ValueError(f"Episode identity/checkpoint mismatch in {path}, row {number}")
    return rows


def _append_rows(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _load_episode_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(values: Iterable[float]) -> tuple[float, float]:
    data = list(values)
    if not data:
        return float("nan"), float("nan")
    return statistics.mean(data), statistics.stdev(data) if len(data) > 1 else 0.0


def _write_summary(path: Path, rows: list[dict[str, str]], keys: tuple[str, ...]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = (
        "real_reward", "environment_steps", "realized_goal_guide_return",
        "real_reward_plus_realized_goal_guide", "mean_plan_score",
        "mean_planning_latency_ms", "wm_real_pose_match_rate",
        "wm_real_inventory_match_rate",
    )
    mean_fields = tuple(
        metric if metric.startswith("mean_") else f"mean_{metric}"
        for metric in metrics
    )
    std_fields = tuple(
        f"std_{metric[5:]}" if metric.startswith("mean_") else f"std_{metric}"
        for metric in metrics
    )
    summary_rows = []
    for key, group in sorted(groups.items()):
        output = dict(zip(keys, key))
        successes = [row for row in group if row["success"].lower() == "true"]
        output.update({
            "episodes": len(group),
            "n_models": len({row["model_id"] for row in group}),
            "n_targets": len({row["target_id"] for row in group}),
            "success_rate": len(successes) / len(group),
            "successful_episode_count": len(successes),
        })
        for metric, mean_field, std_field in zip(metrics, mean_fields, std_fields):
            values = [float(row[metric]) for row in group]
            mean, std = _stats(values)
            output[mean_field] = mean
            output[std_field] = std
        success_steps = [float(row["environment_steps"]) for row in successes]
        output["mean_steps_when_successful"] = (
            statistics.mean(success_steps) if success_steps else float("nan")
        )
        summary_rows.append(output)
    summary_fields = keys + (
        "episodes", "n_models", "n_targets", "success_rate", "successful_episode_count",
        *mean_fields, *std_fields, "mean_steps_when_successful",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def _summarize(run_root: Path, rows: list[dict[str, str]]) -> None:
    _write_summary(
        run_root / "model_target_k_summary.csv", rows,
        ("baseline", "model_id", "target_id", "horizon"),
    )
    _write_summary(
        run_root / "baseline_k_summary.csv", rows,
        ("baseline", "horizon"),
    )


def _case_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["baseline"], row["model_id"], row["target_id"], row["horizon"],
        row["eval_seed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV columns: baseline,model_id,checkpoint")
    parser.add_argument("--baselines", default="",
                        help="Comma-separated manifest baseline labels; default runs all")
    parser.add_argument("--targets", default="1,2,3,4,5",
                        help="Target task numbers, default: 1,2,3,4,5")
    parser.add_argument("--k-values", default="8,16,32,64,128",
                        help="Shared H=K values, default: 8,16,32,64,128")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=0,
                        help="Shared environment seed; each episode uses eval_seed + episode")
    parser.add_argument("--max-ep-len", type=int, default=512)
    parser.add_argument("--reward-mode", choices=("native_goal", "legacy_dense"),
                        default="native_goal")
    parser.add_argument("--goal-guide-weight", type=float, default=0.05)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--results-root", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only when run configuration exactly matches")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan without launching MPC")
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_seed < 0 or args.max_ep_len < 1:
        parser.error("episodes/max-ep-len must be positive and eval-seed nonnegative")
    if args.goal_guide_weight < 0 or not math.isfinite(args.goal_guide_weight):
        parser.error("goal-guide-weight must be finite and nonnegative")
    if args.population < 1 or not 1 <= args.elite_count <= args.population:
        parser.error("require population >= 1 and 1 <= elite-count <= population")
    if args.iterations < 1 or args.cpu_threads < 1 or not 0.0 <= args.gamma <= 1.0:
        parser.error("iterations/cpu-threads must be positive and gamma in [0,1]")
    try:
        target_ids = _csv_ints(args.targets, "targets")
        k_values = _csv_ints(args.k_values, "k-values")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if any(k > args.max_ep_len for k in k_values):
        parser.error("every K/H must be <= max-ep-len")

    models = _read_manifest(args.manifest.expanduser().resolve())
    available_baselines = sorted({model["baseline"] for model in models})
    selected_baselines = (
        [item.strip() for item in args.baselines.split(",") if item.strip()]
        if args.baselines.strip() else available_baselines
    )
    unknown = sorted(set(selected_baselines) - set(available_baselines))
    if unknown:
        parser.error(f"Unknown baselines {unknown}; manifest has {available_baselines}")
    models = [model for model in models if model["baseline"] in selected_baselines]
    if not models:
        parser.error("No model checkpoints selected")
    for model in models:
        checkpoint = Path(model["checkpoint"])
        if not checkpoint.is_file():
            parser.error(
                f"Selected checkpoint does not exist for {model['baseline']}/{model['model_id']}: {checkpoint}"
            )
        model["checkpoint_sha256"] = _sha256(checkpoint)

    targets = {}
    for target_id in target_ids:
        target = f"target_task{target_id}"
        layout = TARGET_ROOT / f"{target}.txt"
        if not layout.is_file():
            parser.error(f"Missing MiniGrid target layout: {layout}")
        targets[target_id] = {"name": target, "layout": str(layout.resolve()),
                              "layout_sha256": _sha256(layout)}

    run_id = args.run_id.strip() or time.strftime("k_sweep_%Y%m%d_%H%M%S")
    try:
        run_id = _safe_component(run_id)
    except ValueError as exc:
        parser.error(str(exc))
    run_root = args.results_root.expanduser().resolve() / run_id

    run_config = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "baselines": selected_baselines,
        "models": models,
        "targets": {str(key): value for key, value in targets.items()},
        "k_values": k_values,
        "horizon_equals_execute_steps": True,
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "max_ep_len": args.max_ep_len,
        "reward_mode": args.reward_mode,
        "goal_guide_weight": args.goal_guide_weight,
        "population": args.population,
        "elite_count": args.elite_count,
        "iterations": args.iterations,
        "gamma": args.gamma,
        "cpu_threads": args.cpu_threads,
    }
    config_path = run_root / "run_config.json"
    if args.dry_run:
        case_count = len(models) * len(targets) * len(k_values)
        print(f"Models: {len(models)} across {len(selected_baselines)} baseline(s)")
        print(f"Targets: {', '.join(v['name'] for v in targets.values())}")
        print(f"H=K: {','.join(map(str, k_values))}")
        print(f"Episodes per case: {args.episodes}; total cases: {case_count}; total episodes: {case_count * args.episodes}")
        print(f"Results: {run_root}")
        first = _command(
            model=models[0], target_id=target_ids[0], horizon=k_values[0],
            eval_seed=args.eval_seed, episodes=args.episodes, max_ep_len=args.max_ep_len,
            reward_mode=args.reward_mode, guide_weight=args.goal_guide_weight,
            population=args.population, elite_count=args.elite_count,
            iterations=args.iterations, gamma=args.gamma, cpu_threads=args.cpu_threads,
            case_root=run_root / "cases" / _safe_component(models[0]["baseline"]) /
                      _safe_component(models[0]["model_id"]) / targets[target_ids[0]]["name"] /
                      f"H{ k_values[0] }_K{ k_values[0]}",
        )
        print("Example command:")
        print(shlex.join(first))
        return

    if run_root.exists():
        if not args.resume:
            parser.error(f"Run directory exists; choose a new --run-id or use --resume: {run_root}")
        if not config_path.is_file():
            parser.error(f"Cannot resume without run_config.json: {run_root}")
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        if prior_config != run_config:
            parser.error(f"Resume settings differ from saved run_config.json: {run_root}")
    else:
        if args.resume:
            parser.error(f"Cannot resume a run directory that does not exist: {run_root}")
        run_root.mkdir(parents=True, exist_ok=False)
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    aggregate_path = run_root / "episode_results.csv"
    if not aggregate_path.exists():
        _append_rows(aggregate_path, [], RESULT_FIELDS)
    all_rows = _load_episode_rows(aggregate_path)
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        grouped.setdefault(_case_key(row), []).append(row)

    case_count = len(models) * len(targets) * len(k_values)
    case_number = 0
    for model in models:
        baseline_component = _safe_component(model["baseline"])
        model_component = _safe_component(model["model_id"])
        checkpoint = Path(model["checkpoint"])
        for target_id in target_ids:
            target_info = targets[target_id]
            for horizon in k_values:
                case_number += 1
                key = (model["baseline"], model["model_id"], str(target_id),
                       str(horizon), str(args.eval_seed))
                existing = grouped.get(key, [])
                if existing:
                    if len(existing) != args.episodes:
                        raise RuntimeError(f"Incomplete aggregate rows for case {key}; inspect {aggregate_path}")
                    if not args.resume:
                        raise RuntimeError(f"Duplicate case in run output: {key}")
                    print(f"[skip {case_number}/{case_count}] already complete: {key}", flush=True)
                    continue

                case_root = (
                    run_root / "cases" / baseline_component / model_component /
                    target_info["name"] / f"H{horizon}_K{horizon}"
                )
                planner_output = case_root / "planner"
                planner_summary = planner_output / "wm_mpc_results.csv"
                case_root.mkdir(parents=True, exist_ok=True)
                if args.resume and planner_summary.is_file():
                    source_rows = _read_results(
                        planner_summary, episodes=args.episodes,
                        eval_seed=args.eval_seed, checkpoint=checkpoint,
                    )
                    print(f"[recover {case_number}/{case_count}] {key}", flush=True)
                else:
                    command = _command(
                        model=model, target_id=target_id, horizon=horizon,
                        eval_seed=args.eval_seed, episodes=args.episodes,
                        max_ep_len=args.max_ep_len, reward_mode=args.reward_mode,
                        guide_weight=args.goal_guide_weight, population=args.population,
                        elite_count=args.elite_count, iterations=args.iterations,
                        gamma=args.gamma, cpu_threads=args.cpu_threads,
                        case_root=case_root,
                    )
                    log_path = case_root / "launcher.log"
                    print(f"[run {case_number}/{case_count}] {key}", flush=True)
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        log_handle.write(shlex.join(command) + "\n\n")
                        log_handle.flush()
                        result = subprocess.run(
                            command, cwd=WM_ROOT, env=_child_env(), stdout=log_handle,
                            stderr=subprocess.STDOUT, text=True, check=False,
                        )
                    if result.returncode != 0:
                        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                        print("\n".join(tail), file=sys.stderr)
                        raise SystemExit(
                            f"MPC failed with exit code {result.returncode}; see {log_path}"
                        )
                    source_rows = _read_results(
                        planner_summary, episodes=args.episodes,
                        eval_seed=args.eval_seed, checkpoint=checkpoint,
                    )

                enriched = []
                for row in source_rows:
                    episode = int(row["episode"])
                    enriched.append({
                        "baseline": model["baseline"],
                        "model_id": model["model_id"],
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": model["checkpoint_sha256"],
                        "target_id": target_id,
                        "target": target_info["name"],
                        "horizon": horizon,
                        "execute_steps": horizon,
                        "eval_seed": args.eval_seed,
                        **{field: row[field] for field in PLANNER_FIELDS},
                    })
                _append_rows(aggregate_path, enriched, RESULT_FIELDS)
                all_rows.extend(enriched)
                grouped[key] = enriched
                _summarize(run_root, all_rows)
                print(f"[done {case_number}/{case_count}] {planner_summary}", flush=True)

    _summarize(run_root, all_rows)
    print(f"Episode results: {aggregate_path}")
    print(f"Model/target/K summary: {run_root / 'model_target_k_summary.csv'}")
    print(f"Baseline/K summary: {run_root / 'baseline_k_summary.csv'}")


if __name__ == "__main__":
    main()
