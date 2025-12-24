
from trainer.common.utils import set_seed
from modelBased.data.data_collect import collect_data_general, visualize_saved_dataset
from modelBased.common.utils import TRAINER_PATH
import os


def collect_data_for_txt(cfg):
    """
    Collects data from a MiniGrid environment defined by a text file.

    Args:
        cfg (DictConfig): Configuration object containing environment and collection settings
    """
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)

    env_text_file_name = '3obstacles_target_task.txt'
    file_name = os.path.splitext(env_text_file_name)[0]
    explore_type = cfg.env.collect.data_type  # 'random' or 'uniform'
    cfg.env.collect.data_save_path = TRAINER_PATH / 'data' / f'{file_name}_test_{explore_type}.npz'
    cfg.env.collect.visualize_save_path = TRAINER_PATH / 'logs' / 'dataset_visualization'
    cfg.env.collect.visualize_filename = f"{file_name}_{explore_type}.png"

    collect_data_general(
        cfg,
        env_source=TRAINER_PATH / 'level' / env_text_file_name,
        save_name=file_name,
        max_steps=1000
    )

def visualize_CL_dataset(cfg):
    """
    Visualizes the agent coverage from a saved dataset.

    Args:
        data_path (str): Path to the saved dataset (.npz file).
        save_path (str): Directory to save the visualization.
        fig_name (str): Filename for the saved figure.
    """
    seed = getattr(cfg, "seed", 0)
    set_seed(seed)
    data_save_dir = TRAINER_PATH / 'data'
    env_text_file_name = '3obstacles_target_task.txt'
    file_name = os.path.splitext(env_text_file_name)[0]
    explore_type = cfg.env.collect.data_type  # 'random' or 'uniform'
    cfg.env.collect.data_save_path = os.path.join(data_save_dir, f'{file_name}_test_{explore_type}.npz')
    cfg.env.collect.visualize_save_path = TRAINER_PATH / 'logs' / 'dataset_visualization'
    cfg.env.collect.visualize_filename = f"{file_name}_{explore_type}.png"

    data_path = cfg.env.collect.data_save_path
    save_path = cfg.env.collect.visualize_save_path
    fig_name = cfg.env.collect.visualize_filename

    visualize_saved_dataset(
        data_path=data_path,
        save_path=os.path.join(save_path, fig_name),
        fig_name=fig_name
    )