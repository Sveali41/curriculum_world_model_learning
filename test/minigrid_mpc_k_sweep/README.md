# MiniGrid MPC H=K sweep

This launcher compares frozen WM checkpoints under identical MiniGrid MPC settings. It fixes `H=K` to `8,16,32,64,128` by default and evaluates target maps 1–5. It runs 10 episodes per model/target/K case by default, matching the MiniGrid evaluation protocol; set `--episodes 1` for a smoke run. The default planner objective is `native_goal` with guide weight `0.05`; real environment reward and realized guide reward are recorded separately by the planner.

## Model manifest

`models.csv` is prefilled with the local MiniGrid WM checkpoint paths for MAC, DR, Target, and P2E, seeds 0–4. Edit the manifest if you want different checkpoints. Its columns are:

```csv
baseline,model_id,checkpoint
mac,seed0,wm/modelBased/models/AttentionWM/mac_attention_world_model_minigrid_none_effect_seed0.ckpt
dr,seed0,wm/modelBased/models/AttentionWM/dr_attention_world_model_minigrid_none_effect_seed0.ckpt
```

Checkpoint paths can be absolute or relative to the project root. `model_id` should distinguish checkpoints, for example `seed0` through `seed4`.

## Run

From the project root, preview one baseline first:

```bash
python3 test/minigrid_mpc_k_sweep/run_k_sweep.py \
  --manifest test/minigrid_mpc_k_sweep/models.csv \
  --baselines mac \
  --run-id mac_hk_5targets \
  --dry-run
```

Then run it:

```bash
python3 test/minigrid_mpc_k_sweep/run_k_sweep.py \
  --manifest test/minigrid_mpc_k_sweep/models.csv \
  --baselines mac \
  --run-id mac_hk_5targets
```

Use a different `--baselines` and `--run-id` for DR, Target, or P2E. Or put all baselines in the manifest and pass a comma-separated list. Each checkpoint group runs in one persistent worker, which loads the WM once and reuses it across its target/K cases; every case still has a separate directory. `--workers` controls concurrent checkpoint groups (default 2), and the parent alone writes aggregate CSV rows in fixed manifest/target/K order. On the tested 8 GB GPU, `--workers 2` passed the throughput, memory, and trajectory checks; one-checkpoint sweeps naturally use only one worker. One baseline with five WM checkpoints, five targets, five K values, and 10 episodes per case runs 1,250 episodes. To do a small check, narrow the sweep, for example `--targets 2 --k-values 8 --episodes 1 --max-ep-len 64 --population 16 --elite-count 4 --iterations 1`.

Add `--resume` with the same run ID to continue a stopped run. The launcher checks the saved experiment configuration and checkpoint hashes before resuming completed cases; worker count may change because it only changes scheduling. Console output defaults to every 100 real steps while all step rows remain in the CSV. Add `--capture-action-hashes` for a parity audit (candidate sequence, ranking, and selected action-list SHA-256); it adds overhead and should be disabled for timing runs. Add `--profile-timing` for synchronized per-episode timing and peak CUDA memory; profiling adds overhead. The sweep uses the attention-weights-off path by default; fixed-seed checks at K=8, 16, 32, 64, and 128 matched candidate sequences, candidate rankings, selected action lists, and real trajectories exactly. Use `--return-attention-weights` to select the original path.

## Outputs

Results live under `test/minigrid_mpc_k_sweep/results/<run-id>/`:

- `episode_results.csv`: per-episode native reward, realized guide return, combined diagnostic, success, steps, planner scores, WM match rates, and case wall time.
- `wm_mpc_action_hashes.csv` (when enabled): CEM candidate sequence, candidate ranking, and selected full action-list hashes by plan.
- `wm_mpc_timing.csv` (when enabled): model loading, total planning, WM forward/decode, remaining planning, environment stepping, logging, and peak allocated CUDA memory.
- `worker_timing.csv`: per-checkpoint worker process/model-load time and summed case time.
- `model_target_k_summary.csv`: summaries per baseline, checkpoint, target, and K.
- `baseline_k_summary.csv`: aggregate comparison per baseline and K.
- `cases/<baseline>/<model>/<target>/H<K>_K<K>/planner/`: full per-step, per-block, and trace CSV/JSON for each case.
- `run_config.json`: exact settings and checkpoint/map SHA-256 hashes.
