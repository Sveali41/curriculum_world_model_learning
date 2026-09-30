#!/usr/bin/env python3
"""Run same-checkpoint MiniGrid MPC sweep cases in one model-owning process."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WM_ROOT = ROOT / "wm"
CONFIG_DIR = WM_ROOT / "modelBased" / "config"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))
    if not jobs:
        raise ValueError("Worker received an empty case list")

    sys.path.insert(0, str(WM_ROOT))
    from hydra import compose, initialize_config_dir
    from modelBased.policy_training.planners import mpc_planner

    worker_started = time.perf_counter()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        first_cfg = compose(config_name="config", overrides=jobs[0]["overrides"])
        checkpoint = Path(str(first_cfg.PPO.checkpoint_path_wm)).expanduser().resolve()
        reuse_loaded_model = mpc_planner.DEVICE.type == "cuda"
        if reuse_loaded_model:
            load_started = time.perf_counter()
            loaded_model = mpc_planner._load_world_model(first_cfg)
            model_load_seconds = time.perf_counter() - load_started
        else:
            loaded_model = None
            model_load_seconds = 0.0

        for job in jobs:
            cfg = compose(config_name="config", overrides=job["overrides"])
            configured_checkpoint = Path(
                str(cfg.PPO.checkpoint_path_wm)
            ).expanduser().resolve()
            if configured_checkpoint != checkpoint:
                raise ValueError(
                    "A worker can only reuse one checkpoint: "
                    f"{configured_checkpoint} != {checkpoint}"
                )
            log_path = Path(job["log_path"])
            runtime_path = Path(job["runtime_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            case_started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write("Reused model checkpoint: " + str(checkpoint) + "\n\n")
                log_handle.flush()
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    mpc_planner.run_online_mpc(
                        cfg, loaded_model=loaded_model if reuse_loaded_model else None
                    )
            case_seconds = time.perf_counter() - case_started
            runtime_path.write_text(
                json.dumps({"case_wall_seconds": case_seconds}, indent=2) + "\n",
                encoding="utf-8",
            )

    worker_seconds = time.perf_counter() - worker_started
    worker_metrics_path = Path(jobs[0]["worker_metrics_path"])
    worker_metrics_path.write_text(
        json.dumps({
            "model_id": jobs[0]["model_id"],
            "checkpoint": str(checkpoint),
            "case_count": len(jobs),
            "model_load_seconds": model_load_seconds,
            "model_reused_across_cases": reuse_loaded_model,
            "worker_seconds": worker_seconds,
            "case_seconds_sum": sum(
                json.loads(Path(job["runtime_path"]).read_text(encoding="utf-8"))["case_wall_seconds"]
                for job in jobs
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[worker] model={jobs[0]['model_id']} cases={len(jobs)} "
          f"load={model_load_seconds:.3f}s worker={worker_seconds:.3f}s", flush=True)


if __name__ == "__main__":
    main()
