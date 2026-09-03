"""Launch one isolated 50x8 MAC run that exports the explorer A/B corpus."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MAP_COUNT = 400
ITERATIONS = 50
BATCH_SIZE = 8
WM_EPOCHS = 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate a neutral, immutable 400-map MAC MiniGrid corpus."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mac-seed", type=int, default=0)
    args = parser.parse_args()

    corpus = args.corpus.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if corpus.exists():
        raise FileExistsError(f"refusing to overwrite corpus: {corpus}")
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse isolated MAC run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    trainer_path = Path(__file__).resolve().parent / "mac_wm_learning.py"
    command = [
        sys.executable,
        str(trainer_path),
        f"seed={args.mac_seed}",
        f"generator_agent.total_iterations={ITERATIONS}",
        f"domains.minigrid.generator_batch_size={BATCH_SIZE}",
        "domains.minigrid.exploration_policy=random",
        f"domains.minigrid.explorer_ab.expected_corpus_size={MAP_COUNT}",
        f"domains.minigrid.explorer_ab.corpus_export_path={corpus}",
        f"attention_model.n_epochs={WM_EPOCHS}",
        "domains.minigrid.val_n_phases=0",
        f"attention_model.model_save_path={run_dir / 'world_model.ckpt'}",
        f"mac_results_dir={run_dir / 'results'}",
        f"mac_temp_data_dir={run_dir / 'temporary_data'}",
        f"hydra.run.dir={run_dir / 'hydra'}",
        "force_fresh_start=true",
        "env.collect.save_env_visualize=false",
        "env.collect.save_coverage_visualize=false",
    ]
    print("[Explorer A/B] launching isolated MAC corpus generation")
    subprocess.run(command, check=True, cwd=trainer_path.parents[1])
    if not corpus.exists():
        raise RuntimeError(f"MAC run completed without producing corpus: {corpus}")
    print(f"[Explorer A/B] corpus written to {corpus}")


if __name__ == "__main__":
    main()
