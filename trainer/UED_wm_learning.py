import hydra
from omegaconf import DictConfig
import os
import torch
import numpy as np

from modelBased.common.utils import TRAINER_PATH
from modelBased.world_model import AttentionWM_training
from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.continue_learning.fisher_buffer import FisherReplayBuffer

from generator.generator_interface import GeneratorInterface
from trainer.common.utils import (
    set_seed,
    validate_on_target_task,
    save_validation_csv,
    convert_trajectories_to_batch,
)


@hydra.main(
    version_base=None,
    config_path=str(TRAINER_PATH / "conf"),
    config_name="config_UED",
)
def adversarial_ued_training(cfg: DictConfig):
    """
    UED Adversarial Training Loop.
    Integrates Generator (PPO), World Model (AttentionWM), and Continual Learning (Fisher Buffer).
    """

    # --------------------------------------
    # 1. 设置与初始化
    # --------------------------------------
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志与数据路径
    log_dir = TRAINER_PATH / "logs" / "results"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = log_dir / "ued_adversarial_log.csv"
    data_save_dir = TRAINER_PATH / "data"

    # === A. 初始化 World Model ===
    wm_instance = AttentionWorldModel(cfg.attention_model).to(device)

    # === B. 初始化 Generator Interface ===
    gen_interface = GeneratorInterface(
        world_model=wm_instance,
        device=device,
        cfg=cfg,
    )

    # === C. 初始化 Fisher Replay Buffer ===
    fisher_buffer = FisherReplayBuffer(
        max_size=cfg.attention_model.fisher_buffer_size
    )

    # === D. 训练状态变量 ===
    old_params, fisher = None, None
    total_iterations = cfg.generator_agent.total_iterations
    wm_train_frequency = cfg.generator_agent.wm_train_frequency  


    # === E. 验证集定义 ===
    target_tasks = [
        "target_task.txt",
        "target_task1.txt",
        "target_task2.txt",
    ]

    target_files = [
        os.path.splitext(t)[0] + "_test_uniform.npz"
        for t in target_tasks
    ]

    print(
        f">>> Starting UED Adversarial Training for {total_iterations} iterations..."
    )

    # --------------------------------------
    # 2. 主循环 (The Loop)
    # --------------------------------------
    for iteration in range(total_iterations):
        print(
            f"\n=== Iteration {iteration + 1}/{total_iterations} ==="
        )

        # --------------------------------------------------------
        # Step 1: Generator 运行 (生成 -> 探索 -> 收集)
        # --------------------------------------------------------
        print(
            "[Generator] Generating environments and collecting trajectories..."
        )

        valid_trajectories = gen_interface.step()

        num_valid_trajs = len(valid_trajectories)
        print(
            f"[Generator] Collected {num_valid_trajs} valid trajectories."
        )

        if num_valid_trajs == 0:
            print(
                "[Warning] No valid trajectories this round. Skipping updates."
            )
            continue

        # --------------------------------------------------------
        # Step 2: 处理数据并存入 Fisher Buffer (WM 的训练数据)
        # --------------------------------------------------------
        new_batch = convert_trajectories_to_batch(valid_trajectories)

        if new_batch is not None:
            # FIX: 使用标准 key，而不是 a/b/c
            buffer_input = {
                "obs": new_batch["obs"],        # FIX
                "obs_next": new_batch["obs_next"],  # FIX
                "act": new_batch["act"],        # FIX
                "info": None,
            }

            fisher_buffer.update_combined(
                buffer_input,
                cfg.attention_model.current_sample_ratio,
                cfg.attention_model.fisher_buffer_elements_ratio,
            )
            print(
                f"[Buffer] Added {len(new_batch['obs'])} transitions. "
                f"Buffer Size: {len(fisher_buffer)}"
            )

        # --------------------------------------------------------
        # Step 3: 更新 Generator (PPO Update)
        # --------------------------------------------------------
        gen_loss = gen_interface.update()

        # FIX: 防止 gen_loss 为 None 时格式化报错
        if gen_loss is not None:
            print(
                f"[Generator] Policy Updated. Loss: {gen_loss:.4f}"
            )
        else:
            print("[Generator] Policy Updated.")

        # --------------------------------------------------------
        # Step 4: 更新 World Model (Adversarial Learning)
        # --------------------------------------------------------
        if (
            len(fisher_buffer) > 1000
            and (iteration + 1) % wm_train_frequency == 0  # FIX: 语义一致
        ):
            print("[World Model] Training...")

            replay_data = fisher_buffer.export_dict()

            # FIX: 保护 cfg 不被永久污染
            old_freeze = cfg.attention_model.freeze_weight
            cfg.attention_model.freeze_weight = False

            old_params, fisher = AttentionWM_training.train_api(
                cfg,
                old_params,
                fisher,
                replay_data=replay_data,
            )

            cfg.attention_model.freeze_weight = old_freeze

            # FIX: 安全加载参数（state_dict / module）
            if isinstance(old_params, dict):
                wm_instance.load_state_dict(old_params)
                gen_interface.sync_world_model(old_params)
            else:
                wm_instance.load_state_dict(old_params.state_dict())
                gen_interface.sync_world_model(
                    old_params.state_dict()
                )

            print(
                "[System] World Model updated and synced to Generator."
            )

        # --------------------------------------------------------
        # Step 5: 验证与日志 (Validation)
        # --------------------------------------------------------
        if (iteration + 1) % 5 == 0:
            print("\n>>> Validating on Target Tasks...")
            avg_losses = []

            for t_name, t_file in zip(
                target_tasks, target_files
            ):
                loss = validate_on_target_task(
                    cfg,
                    net=wm_instance,
                    old_params=(
                        old_params
                        if old_params is not None
                        else wm_instance.state_dict()
                    ),  # FIX
                    data_save_dir=data_save_dir,
                    target_file=t_file,
                    phase_name=f"Iter_{iteration}",
                    VALID_TIMES=1,
                )
                avg_losses.append(loss)
                print(
                    f"Task {t_name}: Loss = {loss:.5f}"
                )

            mean_loss = np.mean(avg_losses)

            save_validation_csv(
                csv_path=csv_path,
                seed=seed,
                mode="UED_Adversarial",
                phase_name=f"Iter_{iteration}",
                transitions=len(fisher_buffer),
                loss=mean_loss,
            )
            print(
                f"[Validation] Mean Loss: {mean_loss:.5f} saved to csv."
            )

    print(">>> UED Adversarial Training Finished.")


if __name__ == "__main__":
    adversarial_ued_training()
