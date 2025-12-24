from omegaconf import DictConfig
from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model import AttentionWM_training
import hydra
import os
from trainer.common.utils import set_seed,split_targets_into_minitasks, \
    create_data_subsets, train_wm_with_subsets, validate_on_target_task, save_validation_csv, \
    extract_loss_map_over_validations, plot_loss_heatmap, collect_data_general


'''
Process
1. load the minitasks
2. collect data from the env
3. train(finetuning) the attention & WM
4. using the trained attention & WM to play in the final task 
5. return score in the final task as the feedback
'''




@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def test_1(cfg: DictConfig):
    """
    Performs continual training of the Attention-based World Model (WM) on a sequence of environments.
    Optionally supports interval training and validation:
      - If interval_train_steps is None, behaves as before (train once per minitask, then validate VALID_TIMES times).
      - If interval_train_steps is an integer, trains for 'interval_train_steps' and validates 'validation_rounds' times.
    """
    import csv
    import numpy as np
    from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
    from modelBased.world_model.AttentionWM import AttentionWorldModel
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    # ----------------------------------------------------------
    # New optional variables for interval control
    # ----------------------------------------------------------
    interval_train_steps = None     # e.g. 1000 to enable interval mode, None = original behavior
    validation_rounds = 10          # number of validation rounds if interval mode enabled
    VALID_TIMES = 50                # number of target validations per phase (original setting)

    csv_path = os.path.join(TRAINER_PATH, 'logs', 'results', 'target_eval_log.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    old_params, fisher = None, None

    env_text_file_name = [
        'only_wall_minitask.txt',
        'only_wall_minitask_2.txt',
        'only_lava_minitask.txt',
        'only_lava_minitask_2.txt',
        'only_key_minitask.txt',
        'only_key_minitask_2.txt',
        'only_door_minitask.txt',
        'only_door_minitask_2.txt',
    ]
    step_len = len(env_text_file_name)
    data_save_dir = TRAINER_PATH / 'data'

    print(f"Saving target validation results to CSV: {csv_path}")

    for step in range(step_len):
        print(f"\n===== Training Phase {step+1}/{step_len}: {env_text_file_name[step]} =====")
        cfg.attention_model.freeze_weight = False

        file_name = os.path.splitext(env_text_file_name[step])[0]
        phase_name = file_name
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{file_name}_test_uniform.npz')

        if len(fisher_buffer) > 0:
            replay_data = fisher_buffer.export_dict()
            print(f"Using replay data with {len(fisher_buffer)} samples.")
        else:
            replay_data = None

        # ----------------------------------------------------------
        # (1) Training logic
        # ----------------------------------------------------------
        if interval_train_steps is None:
            # Original full-phase training
            cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher, replay_data=replay_data)
            old_params, fisher = cur_old_params, cur_fisher

        else:
            # Interval training mode: multiple partial trainings before validation
            for round_idx in range(validation_rounds):
                print(f"[{phase_name}] Interval training {round_idx+1}/{validation_rounds} "
                      f"({interval_train_steps} steps each)")
                cur_old_params, cur_fisher = AttentionWM_training.train_api(
                    cfg, old_params, fisher, replay_data=replay_data, train_steps=interval_train_steps
                )
                old_params, fisher = cur_old_params, cur_fisher

        # ----------------------------------------------------------
        # (2) Update Fisher replay buffer
        # ----------------------------------------------------------
        task_npz = np.load(cfg.attention_model.data_dir, allow_pickle=True)
        samples = {
            'obs': task_npz['a'],
            'obs_next': task_npz['b'],
            'act': task_npz['c'],
            'info': task_npz['f'] if 'f' in task_npz else None
        }
        model_eval = AttentionWorldModel(cfg.attention_model).to(device)
        fisher_buffer.update_combined(samples, 0.3, 0.5)

        # ----------------------------------------------------------
        # (3) Validation (same as before)
        # ----------------------------------------------------------
        print(f"\nStart validating target task {VALID_TIMES} times for phase: {phase_name}")
        results_to_save = []

        cfg.attention_model.freeze_weight = True
        target_file = '3obstacles_target_task_test_uniform.npz'
        cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)
        cfg.attention_model.keep_cell_loss = True
        
        avg_loss_map, loss_list = extract_loss_map_over_validations(
                                        cfg,
                                        old_params=old_params,
                                        data_dir=os.path.join(data_save_dir, target_file),
                                        valid_times=VALID_TIMES
                                    )


        # ----------------------------------------------------------
        # (5) Plot average loss heatmap (unchanged)
        # ----------------------------------------------------------
        output_path = TRAINER_PATH / 'logs' / f"loss_map_avg_{phase_name}.png"
        plot_loss_heatmap(
            loss_map=avg_loss_map,
            save_path=str(output_path),
            phase_name=phase_name
        )



@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def curriculum_learning_transitions(cfg: DictConfig):
    """
    Curriculum Learning (CL) with Fisher Replay Buffer.
    Using modular functions:
        - create_data_subsets()
        - train_wm_with_subsets()
        - validate_on_target_task()
        - save_validation_csv()
    """

    import numpy as np
    import torch
    import os
    import gc
    import matplotlib.pyplot as plt

    from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
    from modelBased.world_model.AttentionWM import AttentionWorldModel

    # --------------------------------------
    # Setup
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    test = False # True: using random data directly collect from target task -- baseline / False: using minitask strings
    manually_define_minitask_name = True # True: manually define minitask names / False: auto generate minitask names
    interval_size = 4266 # number of transitions per training phase # not split when None # 3125 for each of baseline
    training_data_intotal = None # total number of transitions for training
    explore_type = cfg.env.collect.data_type # uniform / random
    data_save_dir = TRAINER_PATH / "data"

    log_dir = TRAINER_PATH / "logs"/ "results"
    csv_path = log_dir / "target_eval_log_compare_6_targets.csv"
    os.makedirs(log_dir, exist_ok=True)

    target_task_name = ['target_task.txt', 'target_task1.txt',
                        'target_task2.txt', 'target_task3.txt', 'target_task4.txt', 'target_task5.txt'
                       ]
    if manually_define_minitask_name:
        minitask_name = [ 'combination_minitask_0.txt','combination_minitask_1.txt',
                            'combination_minitask_2.txt', 'combination_minitask_3.txt',
                            'combination_minitask_4.txt', 'combination_minitask_5.txt',
                            'combination_minitask_6.txt', 'combination_minitask_7.txt',
                        
                         ] # manually define minitask names if needed

    target_file = [os.path.splitext(i)[0] + f"_test_uniform.npz" for i in target_task_name]

    fisher_buffer = FisherReplayBuffer(max_size=cfg.attention_model.fisher_buffer_size)
    current_sample_ratio = cfg.attention_model.current_sample_ratio
    fisher_buffer_elements_ratio = cfg.attention_model.fisher_buffer_elements_ratio
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AttentionWorldModel(cfg.attention_model).to(device)
    old_params, fisher = None, None
    transitions_used_total = 0

    # --------------------------------------
    # New Parameter & Setup for N-phase collection
    # --------------------------------------
    N_PHASES_TO_COLLECT = cfg.attention_model.n_phases_to_collect # how many phases to accumulate before training
    
    # Initialize data accumulation buffer
    # Standard keys for transitions: 'obs', 'act', 'obs_next', 'reward', 'terminal', 'info'
    combined_data = {k: [] for k in ['a', 'b', 'c', 'd', 'e', 'f']}
    phases_collected = 0
    if test:
        phase_files = ['target_task.txt',  'target_task1.txt', 'target_task2.txt', 'target_task3.txt','target_task4.txt',
                        'target_task5.txt'
                       ]
        mode = "Baseline" # 'CL' / 'Baseline'
    else:
        if manually_define_minitask_name:
            phase_files = minitask_name
            mode = "CL" # 'CL' / 'Baseline'
        else:
            phase_files = split_targets_into_minitasks(target_task_name, patch_size=3, patches_per_minitask=1, trials=1, add_agent_start=True)
            mode = "CL" # 'CL' / 'Baseline'

    for idx, phase in enumerate(phase_files):
        print(f"\n===== Data Collection: Phase {idx+1}/{len(phase_files)} =====")
        # --------------------------------------
        # Select sources
        # --------------------------------------


        # --------------------------------------
        # Training Loop
        # --------------------------------------
        # 2) minitask string mode (generating)
        # the flag of are we limit the maximum step for exploration
        if training_data_intotal is not None:
            maximum_dataset_size = training_data_intotal // len(phase_files) 
            print(f"maximum_dataset_size per phase: {maximum_dataset_size}")
        else:
            maximum_dataset_size = None

        if test or manually_define_minitask_name:
            # 1) txt file mode (loading)
            phase_name = os.path.splitext(phase)[0]
            dataset_path = collect_data_general(
                cfg,
                env_source=TRAINER_PATH / "level" / phase,
                save_name=phase_name,
                max_steps=100000,
                maximum_dataset_size=300000,
                recollect_data=False
            )
            task_npz = np.load(dataset_path, allow_pickle=True)

        else:
            if not manually_define_minitask_name:
                layout_str, color_str = phase.split("\n\n")
                save_name = f"minitask_{idx}" 
                dataset_path = collect_data_general(
                    cfg,
                    env_source=(layout_str, color_str),
                    save_name=save_name,
                    max_steps=50000,
                    maximum_dataset_size=maximum_dataset_size,
                    recollect_data=False
                )
                phase_name = save_name
                task_npz = np.load(dataset_path, allow_pickle=True)
    # ---------------------------------------------------------------------------------

    # 1. Accumulate collected data
        data_dict = dict(task_npz)
        for k in combined_data.keys():
            if k in data_dict:
                # Append the NumPy array from the current phase to the list for this key
                combined_data[k].append(data_dict[k])
        phases_collected += 1
        
        # 2. Check if training should be triggered
        is_accumulation_complete = (phases_collected >= N_PHASES_TO_COLLECT)
        is_last_phase = (idx == len(phase_files) - 1)
        
        # Only train if we have collected N phases, OR if we're at the very end
        if is_accumulation_complete or (is_last_phase and phases_collected > 0):
            
            # --- Combine collected data into one structure ---
            print(f"--- Training on combined data from last {phases_collected} phases ---")
            
            final_npz = {}
            # Concatenate all lists of arrays into single NumPy arrays for training
            for k, arrays in combined_data.items():
                if arrays: 
                    final_npz[k] = np.concatenate(arrays, axis=0)
            

            # Use a descriptive name for logging the combined phase
            log_phase_name = f"Combined_P{idx - phases_collected + 2}_to_P{idx+1}" 
            
            # --------------------------------------
            # Create subsets (using the combined data)
            # --------------------------------------
            subsets = create_data_subsets(final_npz, interval_size)

            # --------------------------------------
            # Train WM on subsets
            # --------------------------------------
            net, old_params, fisher, phase_transitions_used = train_wm_with_subsets(
                cfg,
                net,
                subsets,
                fisher_buffer,
                temp_dir=TRAINER_PATH /'data'/"temp",
                num_iterations=1,
                old_params=old_params,    
                fisher=fisher,             
                current_sample_ratio=current_sample_ratio,
                fisher_buffer_elements_ratio=fisher_buffer_elements_ratio
            )

            # Determine logging info (transitions and phase name)
            transitions_used_total += phase_transitions_used
            # --------------------------------------
            # VALIDATION (unified)
            # --------------------------------------
            validate_loss_on_target_task = []
            for i in target_file:
                avg_loss = validate_on_target_task(
                    cfg,
                    net,
                    old_params=old_params,
                    data_save_dir=data_save_dir,
                    target_file=i,
                    phase_name=log_phase_name, # Use combined name
                    VALID_TIMES=1
                )
                validate_loss_on_target_task.append(avg_loss)
            avg_loss = float(np.mean(validate_loss_on_target_task))
            print(f"[Unified Validation] {log_phase_name} → Overall Avg Target Loss = {avg_loss:.5f}")

            # --------------------------------------
            # Save CSV (unified)
            # --------------------------------------
            save_validation_csv(
                csv_path=csv_path,
                seed=seed,
                mode=mode,
                phase_name=log_phase_name, # Use combined name
                transitions=transitions_used_total,
                loss=avg_loss
            )
            
            # 3. Reset buffer and counter for the next batch
            combined_data = {k: [] for k in combined_data.keys()}
            phases_collected = 0
            
        else:
            # Continue collecting data if N phases haven't been reached
            print(f"Current accumulation: {phases_collected}/{N_PHASES_TO_COLLECT}. Continuing collection...")
    



@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_CL")
def train_on_target_task_only(cfg: DictConfig):
    """
    Train Attention WM only on the target task dataset.
    """

    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    data_save_dir = TRAINER_PATH / "data"
    target_file = "only_lava_minitask_test_random.npz"

    cfg.attention_model.freeze_weight = False
    cfg.attention_model.data_dir = os.path.join(data_save_dir, target_file)

    old_params, fisher = AttentionWM_training.train_api(cfg, None, None)

if __name__ == "__main__":
    # collect_data_for_txt()
    # count_data_in_dataset("3obstacles_target_task_test.npz")
    # test_1()
    # # visualize_CL_dataset()
    curriculum_learning_transitions()
    # train_on_target_task_only()
