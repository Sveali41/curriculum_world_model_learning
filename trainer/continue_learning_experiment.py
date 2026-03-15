import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from omegaconf import DictConfig
from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model import AttentionWM_training
from datetime import datetime
import hydra
import torch
import numpy as np

from modelBased.policy_training import PPO_world_training


'''
Process
1. load the generator
2. use the generator to generator env 
(comparision among the different env as loss1)
3. collect data from the env
4. train(finetuning) the attention & WM
5. using the trained attention & WM to play in the final task sets
6. return score in the final task as the feedback

'''


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_crafter_CL")
def collect_data(cfg: DictConfig):
    import modelBased.common.support as support_mod
    support = support_mod.Support(cfg)
    
    env_task_names = [
        'crafter_minitask_01', 'crafter_minitask_02', 'crafter_minitask_03', 
        'crafter_minitask_04', 'crafter_minitask_05', 'crafter_minitask_06',
        'crafter_target_task_diamond'
    ]
    level_dir = os.path.join(PROJECT_ROOT, 'trainer', 'level', 'crafter')
    data_save_dir = os.environ.get("TRAIN_DATASET_PATH", os.path.join(PROJECT_ROOT, "modelBased/data/train_world_model"))
    
    # Ensure save directory exists
    os.makedirs(data_save_dir, exist_ok=True)

    for task_name in env_task_names:
        print(f"\n--- Collecting data for {task_name} ---")
        file_path = os.path.join(level_dir, f"{task_name}.txt")
        
        # Wrap environment from text file
        env = support.wrap_env_from_text(file_path, max_steps=20000)
        
        # Set collection path
        cfg.env.collect.data_save_path = os.path.join(data_save_dir, f'{task_name}.npz')
        
        # Execute collection
        support.collect_data_trainer(
            env=env,
            wandb_run=None,
            validate=False,
            save_img=False,
            log_name=f"collect_{task_name}",
            max_steps=20000 # Collect 20k steps per minitask
        )
        print(f"Saved: {cfg.env.collect.data_save_path}")

@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_crafter_CL")
def test_1(cfg: DictConfig):
    """
    Performs continual training of the Attention-based World Model (WM) on a sequence of Crafter tasks.
    Validates on the target uniform dataset after each task training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer
    from modelBased.world_model.AttentionWM import AttentionWorldModel
    import numpy as np

    fisher_buffer = FisherReplayBuffer(max_size=500000)
    old_params, fisher = None, None
    trained_net = None

    # Task sequence based on curriculum
    env_task_names = ['crafter_minitask_01', 'crafter_minitask_02', 'crafter_minitask_03']  
    target_val_dataset = "crafter_target_uniform_test.npz"
    
    data_save_dir = os.environ.get("TRAIN_DATASET_PATH", "/home/siyao/phd_file/Research/rlPractice/Curriculum_world_model_learning/modelBased/data/train_world_model")
    
    for step, task_name in enumerate(env_task_names):
        print(f"\n\n{'#'*60}")
        print(f"### PHASE {step+1}: Training on {task_name}")
        print(f"{'#'*60}\n")
        
        cfg.attention_model.freeze_weight = False

        # === 设置当前任务数据路径 ===
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{task_name}.npz')

        # === 混合 replay ===
        replay_data = fisher_buffer.export_dict() if len(fisher_buffer) > 0 else None

        # === 启动训练（含 EWC） ===
        # 注意：AttentionWM_training.train_api 返回的是 (old_params, fisher, net)
        # 我们将 trained_net 传回，以在同一模型实例上继续训练
        cur_old_params, cur_fisher, trained_net = AttentionWM_training.train_api(
            cfg, 
            net=trained_net, 
            old_params=old_params, 
            fisher=fisher, 
            replay_data=replay_data
        )
        old_params, fisher = cur_old_params, cur_fisher

        # === 立即验证：泛化性能测试 (Target Test Set) ===
        print(f"\n--- VALIDATING Phase {step+1} on Target Uniform Dataset ---")
        cfg.attention_model.freeze_weight = True
        cfg.attention_model.data_dir = os.path.join(data_save_dir, target_val_dataset)
        
        # 使用刚刚训练好的网络进行验证
        AttentionWM_training.run(
            cfg, 
            net=trained_net, 
            old_params=old_params, 
            fisher=fisher, 
            replay_data=None
        )

        # === 更新 Fisher Replay Buffer (使用最新的 data_dir 之前训练时的数据) ===
        task_data_path = os.path.join(data_save_dir, f'{task_name}.npz')
        task_npz = np.load(task_data_path, allow_pickle=True)
        samples = {
            'obs': task_npz['a'],
            'obs_next': task_npz['b'],
            'act': task_npz['c'],
            'info': task_npz['f'] if 'f' in task_npz else None,
            'inv': task_npz['g'] if 'g' in task_npz else None,
            'inv_next': task_npz['h'] if 'h' in task_npz else None,
        }
        # 使用训练后的模型更新 Buffer
        fisher_buffer.update_combined(samples, 0.3, 0.5) 


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def test_2(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_params, fisher = None, None
    env_text_file_name = ['env1_test.txt']
    step_len = len(env_text_file_name)

    for step in range(step_len):
        print(f"Step {step+1} of {step_len}...")
        # env = support.wrap_env(support.generate_env(model))
        file_name = os.path.splitext(env_text_file_name[step])[0]  # 'env1_move'
        data_save_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data'
        cfg.attention_model.data_dir = os.path.join(data_save_dir, f'{file_name}.npz')
        cur_old_params, cur_fisher = AttentionWM_training.train_api(cfg, old_params, fisher)
        old_params, fisher = cur_old_params, cur_fisher

    cfg.attention_model.freeze_weight = True
    cfg.attention_model.data_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data/env1_test.npz'
    AttentionWM_training.train_api(cfg, old_params, fisher)
    cfg.attention_model.data_dir = '/home/siyao/project/rlPractice/MiniGrid/trainer/data/env2_test.npz'
    AttentionWM_training.train_api(cfg, old_params, fisher)

    
if __name__ == "__main__":
    collect_data()
    test_1()
    # test_2()