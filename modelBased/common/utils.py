import os
import sys
from pathlib import Path
from typing import Dict, Optional

import dotenv
import numpy as np
import omegaconf
import torch
import torch.serialization

def map_to_nearest_value_support(tensor, valid_values):
    valid_values = torch.tensor(valid_values, dtype=torch.float32).to(tensor.device)
    tensor = tensor.unsqueeze(-1)  # Add a dimension to compare with valid values
    differences = torch.abs(tensor - valid_values)  # Calculate differences
    indices = torch.argmin(differences, dim=-1)  # Get index of nearest value
    nearest_values = valid_values[indices]  # Get nearest values using indices
    return nearest_values

def extract_masked_state_support(state, agent_position_yx, mask_size):
    """
    state dimensions: (channels, rows, cols)
    """
    channels, rows, cols = state.shape
    y, x = agent_position_yx
    half = mask_size // 2
    margin_data = state[:, 0, 0]
    region = np.tile(margin_data.reshape(channels, 1, 1),
                     (1, mask_size, mask_size))

    src_slice_y = slice(max(y - half, 0), min(y + half + 1, rows))
    src_slice_x = slice(max(x - half, 0), min(x + half + 1, cols))

    dest_slice_y = slice(max(0, half - y), max(0, half - y) + (min(y + half + 1, rows) - max(y - half, 0)))
    dest_slice_x = slice(max(0, half - x), max(0, half - x) + (min(x + half + 1, cols) - max(x - half, 0)))

    # 将 state 中的有效区域复制到预填充区域中
    region[:, dest_slice_y, dest_slice_x] = state[:, src_slice_y, src_slice_x]
    return region


def replace_values(arr, old_values, new_values):
    assert arr.ndim >= 2 and len(old_values) == len(new_values)
    mapping = np.arange(256, dtype=arr.dtype)
    mapping[np.array(old_values, dtype=arr.dtype)] = np.array(new_values, dtype=arr.dtype)
    arr[:, :] = np.take(mapping, arr[:, :])
    return arr


def create_mask(state_shape, agent_position, mask_size):
    mask = np.zeros((state_shape[0], state_shape[1]), dtype=bool)
    y, x = agent_position
    half_size = mask_size // 2
    y_start, y_end = max(0, y - half_size), min(state_shape[0], y + half_size + 1)
    x_start, x_end = max(0, x - half_size), min(state_shape[1], x + half_size + 1)
    mask[y_start:y_end, x_start:x_end] = True
    return mask


def load_model_weight(model, weight_path, freeze=True):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])

        with torch.serialization.safe_globals([omegaconf.dictconfig.DictConfig]):
            checkpoint = torch.load(weight_path, weights_only=False, map_location=device)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(cleaned_state_dict, strict=False)
        model.to(device)
        model.eval()

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        print(f"[Load] Model weights loaded from: {weight_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Weight file not found at {weight_path}") from e
    except KeyError as e:
        raise KeyError(f"Error: missing key in checkpoint -> {e}") from e
    except RuntimeError as e:
        raise RuntimeError(
            f"Model loading RuntimeError: {e}\n"
            "If this is due to PyTorch safe-unpickle restrictions, "
            "ensure add_safe_globals was applied correctly."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}") from e


def normalize_obs(x, obs_norm_values):
    if isinstance(x, np.ndarray):
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        if not torch.is_floating_point(x):
            x = x.to(torch.float32)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch tensor.")

    if x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError(
                "Normalization values must be provided and must match the number of channels in the data."
            )
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[i, :, :] /= max_val
        x = x.reshape(channel, row, col)

    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError(
                "Normalization values must be provided and must match the number of channels in the data."
            )
        bsz, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] /= max_val
        x = x.reshape(bsz, channel, row, col)
    else:
        raise ValueError("Input must be a 3D or 4D array.")

    return x


def denormalize_obj(x, obs_norm_values):
    if x.ndim == 3:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[0]:
            raise ValueError(
                "Normalization values must be provided and must match the number of channels in the data."
            )
        channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[i, :, :] *= max_val
        x = x.reshape(channel, row, col)

    elif x.ndim == 4:
        if obs_norm_values is None or len(obs_norm_values) != x.shape[1]:
            raise ValueError(
                "Normalization values must be provided and must match the number of channels in the data."
            )
        bsz, channel, row, col = x.shape
        for i in range(channel):
            max_val = obs_norm_values[i]
            if max_val != 0:
                x[:, i, :, :] *= max_val
        x = x.reshape(bsz, channel, row, col)
    else:
        raise ValueError("Input must be a 3D or 4D array.")

    return x


def get_env(env_name: str, default: Optional[str] = None) -> str:
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = ".env") -> None:
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def merge_data_dicts(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = {}
    for key in d1.keys():
        if key in d2:
            merged[key] = np.concatenate([d1[key], d2[key]], axis=0)
        else:
            merged[key] = d1[key]
    return merged


load_envs()
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
GENERATOR_PATH: Path = Path(get_env("GENERATOR_PATH"))
TRAINER_PATH: Path = Path(get_env("TRAINER_PATH"))
WORLD_MODEL_PATH: Path = Path(get_env("WORLD_MODEL_PATH"))
sys.path.append(str(PROJECT_ROOT.resolve()))
