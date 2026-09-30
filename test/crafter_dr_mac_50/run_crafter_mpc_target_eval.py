"""Evaluate DR, MAC, Target, and P2E seed 0–4 snapshots with frozen Crafter MPC.

Defaults use the MiniGrid MPC comparison's H/K settings, five targets,
10 paired evaluation episodes, Crafter's 4096 real steps, and CEM 128/16/3. The Crafter
planner keeps its existing native-reward and progress-guidance objective.
Diamond rate means a real episode unlocked ``collect_diamond``. The guided
score is the mean score of selected imagined plans; it includes Crafter native
reward and existing progress guidance. It is not a real-environment dense
return, so native reward is reported separately.

Examples from the repository root:
    python test/crafter_dr_mac_50/run_crafter_mpc_target_eval.py --dry-run
    python test/crafter_dr_mac_50/run_crafter_mpc_target_eval.py
    python test/crafter_dr_mac_50/run_crafter_mpc_target_eval.py --arms dr,mac --seeds 0 --targets 1 --settings 16:8 --episodes 1

All four arms require their final checkpoints first. Use --arms for a subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = ROOT / "wm"
EXPERIMENT = ROOT / "test" / "crafter_dr_mac_50"
RESULTS = EXPERIMENT / "results"
TARGETS = ROOT / "trainer" / "level" / "crafter" / "target_tasks"
ARMS = {
    "dr": ("dr_ewc20_epoch10_seed", 50),
    "mac": ("mac_balanced_ewc20_epoch10_seed", 60),
    "target": ("seed", 20),
    "p2e": ("seed", 100),
}
METRICS = ("diamond", "guided_score", "real_reward", "environment_steps", "imagined_transitions")
GUIDANCE = {
    "gamma": 0.99,
    "crafter_terminal_penalty": 5.0,
    "crafter_resource_guidance_weight": 3.0,
    "crafter_exploration_weight": 1.0,
    "crafter_tool_progress_weight": 0.25,
    "crafter_inventory_progress_weight": 0.25,
    "crafter_require_three_wood_for_first_table": True,
    "crafter_late_stage_guidance": True,
}


def _integers(value: str, name: str) -> list[int]:
    try:
        numbers = list(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as exc:
        raise SystemExit(f"{name} needs comma-separated integers") from exc
    if not numbers or any(number < 0 for number in numbers):
        raise SystemExit(f"{name} needs nonnegative integers")
    return numbers


def _settings(value: str) -> list[tuple[int, int]]:
    try:
        settings = list(dict.fromkeys(tuple(map(int, part.split(":"))) for part in value.split(",")))
    except ValueError as exc:
        raise SystemExit("--settings needs H:K pairs, for example 16:8,40:20,100:50") from exc
    if not settings or any(len(pair) != 2 or not 1 <= pair[1] <= pair[0] for pair in settings):
        raise SystemExit("Each --settings pair needs 1 <= K <= H")
    return settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(arm: str, seed: int) -> tuple[Path, str]:
    prefix, iteration = ARMS[arm]
    checkpoint = RESULTS / arm / f"{prefix}{seed}" / "wm_snapshots" / f"iter_{iteration:03}.ckpt"
    validation = RESULTS / arm / "full20_changed_focal.csv"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {arm} seed {seed} checkpoint: {checkpoint}")
    digest = _sha256(checkpoint)
    if arm in {"dr", "mac"}:
        if not validation.is_file():
            raise FileNotFoundError(f"Missing plotted full20 validation CSV: {validation}")
        with validation.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if int(row["Seed"]) == seed and int(row["WM_Update"]) == 50]
        if len(rows) != 1:
            raise ValueError(f"Expected one seed {seed}, WM update 50 row in {validation}; found {len(rows)}")
        if digest != rows[0]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint differs from plotted full20 validation: {checkpoint}")
    return checkpoint, digest


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(ROOT), "TRAINER_ROOT": str(ROOT), "WM_ROOT": str(WM_ROOT),
        "TRAINER_PATH": str(ROOT / "trainer"), "ENV_PATH": str(ROOT / "trainer" / "level"),
        "WORLD_MODEL_PATH": str(WM_ROOT / "modelBased"),
        "MODEL_FPATH": str(WM_ROOT / "modelBased" / "models"),
        "TRAIN_DATASET_PATH": str(WM_ROOT / "modelBased" / "data" / "train_world_model"),
    })
    return env


def _case_rows(path: Path, *, episodes: int, eval_seed: int, checkpoint: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} episodes in {path}, found {len(rows)}")
    for number, row in enumerate(rows):
        if (int(row["episode"]) != number or int(row["seed"]) != eval_seed + number
                or Path(row["wm_checkpoint"]).resolve() != checkpoint):
            raise ValueError(f"Episode identity/checkpoint mismatch in {path}, row {number}")
    return rows


def _plan_scores(path: Path, episodes: int) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {episode: [] for episode in range(episodes)}
    for row in rows:
        if int(row["plan_step"]) == 1:
            selected[int(row["episode"])].append(float(row["plan_score"]))
    if any(not scores for scores in selected.values()):
        raise ValueError(f"Missing selected plan scores in {path}")
    return {episode: statistics.mean(scores) for episode, scores in selected.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(episode_rows: list[dict], output_root: Path) -> None:
    by_case: dict[tuple, list[dict]] = {}
    for row in episode_rows:
        key = (row["arm"], row["wm_seed"], row["target"], row["horizon"], row["execute_steps"])
        by_case.setdefault(key, []).append(row)
    case_rows = []
    for (arm, seed, target, horizon, execute), rows in sorted(by_case.items()):
        case_rows.append({
            "arm": arm, "wm_seed": seed, "target": target, "horizon": horizon,
            "execute_steps": execute, "episodes": len(rows),
            **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows)
               for metric in METRICS},
        })
    _write_csv(output_root / "case_summary.csv", case_rows)

    by_seed: dict[tuple, list[dict]] = {}
    for row in episode_rows:
        by_seed.setdefault((row["arm"], row["horizon"], row["execute_steps"], row["wm_seed"]), []).append(row)
    seed_rows = []
    for (arm, horizon, execute, seed), rows in sorted(by_seed.items()):
        seed_rows.append({
            "arm": arm, "horizon": horizon, "execute_steps": execute, "wm_seed": seed,
            "targets": len({row["target"] for row in rows}), "episodes": len(rows),
            **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows)
               for metric in METRICS},
        })
    _write_csv(output_root / "seed_summary.csv", seed_rows)

    by_setting: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        by_setting.setdefault((row["arm"], row["horizon"], row["execute_steps"]), []).append(row)
    summary_rows = []
    for (arm, horizon, execute), rows in sorted(by_setting.items()):
        summary = {
            "arm": arm, "horizon": horizon, "execute_steps": execute,
            "wm_seeds": len(rows), "targets": rows[0]["targets"],
            "episodes": sum(row["episodes"] for row in rows),
        }
        for metric in METRICS:
            seed_means = [row[f"mean_{metric}"] for row in rows]
            summary[f"mean_{metric}"] = statistics.mean(seed_means)
            summary[f"seed_std_{metric}"] = statistics.stdev(seed_means) if len(seed_means) > 1 else ""
        summary_rows.append(summary)
    _write_csv(output_root / "comparison_summary.csv", summary_rows)
    settings = sorted({(row["horizon"], row["execute_steps"]) for row in summary_rows})
    lookup = {(row["arm"], row["horizon"], row["execute_steps"]): row for row in summary_rows}
    headers = ["Baseline"] + [
        f"H{h}/K{k} {metric}" for h, k in settings
        for metric in ("Diamond rate", "Guided score", "Env steps", "Imagined steps")
    ]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for arm in ARMS:
        if not any(row["arm"] == arm for row in summary_rows):
            continue
        cells = [arm.upper()]
        for h, k in settings:
            row = lookup.get((arm, h, k))
            cells.extend((
                f"{row['mean_diamond']:.1%}", f"{row['mean_guided_score']:.3f}",
                f"{row['mean_environment_steps']:.1f}", f"{row['mean_imagined_transitions']:.0f}",
            ) if row else ("—",) * 4)
        table.append("| " + " | ".join(cells) + " |")
    selected_targets = sorted({row["target"] for row in episode_rows})
    selected_seeds = sorted({row["wm_seed"] for row in episode_rows})
    (output_root / "comparison_table.md").write_text(
        "\n".join([
            f"# Crafter MPC: targets {selected_targets}, WM seeds {selected_seeds}",
            "",
            "Diamond rate is the fraction of real episodes unlocking collect_diamond. "
            "Guided score is the mean score of selected imagined plans, including "
            "native reward and the existing progress guidance; it is not a realized dense return. "
            "Compare guided scores within the same H/K setting.",
            "",
            *table, "",
        ]), encoding="utf-8",
    )
    for row in summary_rows:
        print(f"H{row['horizon']}/K{row['execute_steps']} {row['arm'].upper()}: "
              f"diamond={row['mean_diamond']:.1%}, "
              f"guided score={row['mean_guided_score']:.3f}, "
              f"native reward={row['mean_real_reward']:.3f}, "
              f"env steps={row['mean_environment_steps']:.1f}, "
              f"imagined steps={row['mean_imagined_transitions']:.0f}", flush=True)


def _run_case(case: dict, *, args: argparse.Namespace, env: dict[str, str]) -> list[dict]:
    """Run or resume one isolated case and return its episode rows."""
    arm, seed, target = case["arm"], case["seed"], case["target"]
    case_dir, config = case["case_dir"], case["config"]
    checkpoint = case["checkpoint"]
    manifest = case_dir / "case_config.json"
    result = case_dir / "wm_mpc_results.csv"
    steps = case_dir / "wm_mpc_steps.csv"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else None
    if previous is not None and previous != config and not args.force:
        raise ValueError(f"Case settings changed: {manifest}; choose another --output-root or --force")
    complete = previous == config and result.is_file() and steps.is_file()
    label = f"H{case['horizon']}/K{case['execute']} {arm} seed{seed} target{target}"
    print(label, flush=True)
    if complete and not args.force:
        print(f"[SKIP] completed case: {label}", flush=True)
    else:
        case_dir.mkdir(parents=True, exist_ok=True)
        manifest.unlink(missing_ok=True)
        with (case_dir / "run.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(case["command"], cwd=WM_ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"MPC exited {completed.returncode} for {label}; see {case_dir / 'run.log'}")

    rows = _case_rows(result, episodes=args.episodes, eval_seed=args.eval_seed, checkpoint=checkpoint)
    plan_scores = _plan_scores(steps, args.episodes)
    manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return [{
        "arm": arm, "wm_seed": seed, "target": target,
        "horizon": case["horizon"], "execute_steps": case["execute"],
        "episode": int(row["episode"]), "eval_seed": int(row["seed"]),
        "diamond": "collect_diamond" in json.loads(row["achievements"]),
        "guided_score": plan_scores[int(row["episode"])],
        "real_reward": float(row["real_reward"]),
        "environment_steps": int(row["environment_steps"]),
        "imagined_transitions": int(row["imagined_transitions"]),
        "achievements": row["achievements"], "checkpoint": str(checkpoint),
        "result_dir": str(case_dir),
    } for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="dr,mac,target,p2e", help="Comma-separated subset of dr,mac,target,p2e")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--targets", default="1,2,3,4,5")
    parser.add_argument("--settings", default="16:8,40:20,100:50", help="Comma-separated H:K pairs")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=4096)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Maximum number of independent MPC cases to run concurrently")
    parser.add_argument("--output-root", type=Path, default=RESULTS / "mpc_crafter_dr_mac_50")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed cases")
    args = parser.parse_args()
    arms = list(dict.fromkeys(part.strip() for part in args.arms.split(",") if part.strip()))
    if not arms or set(arms) - set(ARMS):
        parser.error("--arms must contain dr, mac, target and/or p2e")
    seeds, targets, settings = _integers(args.seeds, "--seeds"), _integers(args.targets, "--targets"), _settings(args.settings)
    if (args.episodes < 1 or args.eval_seed < 0 or args.max_episode_steps < 1
            or args.population < 1 or args.iterations < 1 or args.jobs < 1
            or not 1 <= args.elite_count <= args.population):
        parser.error("Invalid episode, seed, step, population, elite, iteration, or jobs count")
    output_root = args.output_root.expanduser().resolve()
    snapshots = {}
    missing = []
    for arm in arms:
        for seed in seeds:
            try:
                snapshots[(arm, seed)] = _snapshot(arm, seed)
            except FileNotFoundError as exc:
                missing.append(str(exc))
    layouts = {target: TARGETS / f"crafter_target_task_{target}.txt" for target in targets}
    missing.extend(f"Missing target layout: {layout}" for layout in layouts.values() if not layout.is_file())
    if missing:
        raise SystemExit("Preflight failed; train/copy the missing baseline snapshots first, or select available arms:\n"
                         + "\n".join(missing))
    print(f"{len(settings) * len(arms) * len(seeds) * len(targets)} cases; "
          f"{args.episodes} paired episodes each; output={output_root}", flush=True)
    cases = []
    for horizon, execute in settings:
        for arm in arms:
            for seed in seeds:
                checkpoint, checkpoint_sha = snapshots[(arm, seed)]
                for target in targets:
                    layout = layouts[target]
                    case_dir = output_root / f"H{horizon}_K{execute}" / arm / f"seed{seed}" / f"target{target}"
                    config = {
                        "arm": arm, "wm_seed": seed, "target": target, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_sha, "layout": str(layout),
                        "layout_sha256": _sha256(layout), "horizon": horizon, "execute_steps": execute,
                        "episodes": args.episodes, "eval_seed": args.eval_seed,
                        "max_episode_steps": args.max_episode_steps, "population": args.population,
                        "elite_count": args.elite_count, "iterations": args.iterations,
                        "guidance": GUIDANCE,
                    }
                    command = [
                        args.python, "-u", "-m", "modelBased.policy_training.planners.mpc_planner",
                        "domain=crafter", "PPO.wm_control_mode=mpc", "PPO.train_in_real_env=false",
                        f"PPO.seed={args.eval_seed}", f"domains.crafter.task_name=crafter_target_task_{target}",
                        f"PPO.env_path={layout}", f"PPO.mpc.checkpoint_path={checkpoint}",
                        f"PPO.mpc.episodes={args.episodes}", f"PPO.max_ep_len={args.max_episode_steps}",
                        f"PPO.mpc.horizon={horizon}", f"PPO.mpc.execute_steps={execute}",
                        f"PPO.mpc.population={args.population}", f"PPO.mpc.elite_count={args.elite_count}",
                        f"PPO.mpc.iterations={args.iterations}", "PPO.mpc.print_every_steps=512",
                        f"PPO.mpc.crafter_output_dir={case_dir}", f"hydra.run.dir={case_dir / 'hydra'}",
                    ]
                    command.extend(f"PPO.mpc.{name}={str(value).lower()}" for name, value in GUIDANCE.items())
                    if args.dry_run:
                        print(f"H{horizon}/K{execute} {arm} seed{seed} target{target}", flush=True)
                        print("[DRY RUN]", shlex.join(command), flush=True)
                        continue
                    cases.append({
                        "arm": arm, "seed": seed, "target": target,
                        "horizon": horizon, "execute": execute,
                        "checkpoint": checkpoint, "config": config,
                        "case_dir": case_dir, "command": command,
                    })
    if not args.dry_run:
        env = _child_env()
        if args.jobs == 1:
            case_episode_rows = [_run_case(case, args=args, env=env) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                # map yields in input order so the combined CSVs are deterministic.
                case_episode_rows = list(executor.map(
                    lambda case: _run_case(case, args=args, env=env), cases))
        episode_rows = [row for rows in case_episode_rows for row in rows]
        _write_csv(output_root / "episode_results.csv", episode_rows)
        _summaries(episode_rows, output_root)
        print(f"Saved comparison: {output_root / 'comparison_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
