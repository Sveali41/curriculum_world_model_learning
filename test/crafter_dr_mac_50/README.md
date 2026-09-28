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

## Crafter Target and P2E baselines

These separate configs match the DR/MAC WM learning rate, batch size, 10 epochs,
EWC, replay, and fixed target 1–4 validation (500 samples each). They each
collect exactly 400,000 new training transitions; the run fails if a collection
is short. Results and collected training data stay in separate `results/target/`
and `results/p2e/` directories.

```bash
bash test/crafter_dr_mac_50/run_target.sh
bash test/crafter_dr_mac_50/run_p2e.sh
```

Each script runs seed 0 then seed 1 sequentially. Add `--dry-run` to print its
commands without training; extra arguments are forwarded as Hydra overrides.
Target uses 20
updates × 20,000 transitions and P2E uses 100 updates × 4,000 transitions;
DR/MAC use 50 updates × 8,000. The new-data budget is equal, while the number
of WM updates differs by baseline design. Target's full-map observations are
31×31, so its grid setting remains 31×31; DR/MAC and P2E use 8×8 observations.
