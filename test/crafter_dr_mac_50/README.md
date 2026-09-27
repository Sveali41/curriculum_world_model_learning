# Crafter DR vs balanced MAC: 50 WM updates

Both arms use seed 0 or 1, 8,000 new transitions per WM update, 10 WM epochs,
EWC 20, and the same replay and WM configuration. During training, each update
validates uniform targets 1–4 with 500 fixed samples per target. MAC runs 10
novelty warmup rounds before its 50 WM updates. The warmup rounds do not count
as WM updates.

From the repository root, run DR seed 0 then seed 1 on the server:

```bash
bash test/crafter_dr_mac_50/run_dr.sh
```

The scripts forward optional Hydra overrides. For example,
`bash test/crafter_dr_mac_50/run_dr.sh --dry-run` prints both commands without
training.

Run balanced MAC locally, also sequentially for seed 0 and 1:

```bash
bash test/crafter_dr_mac_50/run_mac.sh
```

Run the two seeds sequentially on each machine unless you have measured that
concurrent training fits its GPU memory. Each run has its own results,
checkpoint, and resume state directory in `results/` below this folder.

To resume an interrupted run, repeat its command with
`force_fresh_start=false resume_training=true`. Do not use a different seed or
run ID when resuming.

Each successful WM update saves a checkpoint in its run's `wm_snapshots/`
directory. After selecting an iteration, validate its checkpoint on all 20
uniform targets without retraining:

```bash
python3 -u test/crafter_dr_mac_50/validate_checkpoint.py --config-name config_dr_crafter_dr_mac_50 --iteration 50 seed=0
python3 -u test/crafter_dr_mac_50/validate_checkpoint.py --config-name config_mac_crafter_dr_mac_50 --iteration 60 seed=0
```

Replace the iteration with the selected snapshot number and seed as needed.
DR snapshot 50 and MAC snapshot 60 both represent 50 WM updates. Offline
validation writes `aggregate.csv` and `per_target.csv` under that run's
`offline_validation/` directory.
