import os
from pathlib import Path

from domain.minigrid import minigrid_support
from domain.crafter import crafter_support
from domain.crafter.crafter_custom_env import CustomCrafterEnv
from modelBased.data.data_collect import data_collect_api


class Support:
    def __init__(self, cfg):
        self.cfg = cfg

    def wrap_env(self, env_layout):
        return minigrid_support.wrap_env(env_layout, self.cfg)

    def wrap_env_from_text(self, file_path, max_steps=10000):
        if self.cfg.attention_model.env_type == 'crafter':
            return self.wrap_env_from_text_crafter(file_path, max_steps)
        return minigrid_support.wrap_env_from_text(file_path, max_steps, self.cfg)

    def wrap_env_from_text_crafter(self, file_path, max_steps=10000):
        env = CustomCrafterEnv(txt_file_path=file_path, max_steps=max_steps)
        return env

    def interpret_env(self, env, color_array=None, inventory_vec=None):
        if self.cfg.attention_model.env_type == 'crafter':
            return crafter_support.interpret_env(env, self.cfg, inventory_vec=inventory_vec)
        return minigrid_support.interpret_env(env, self.cfg, color_array=color_array)

    def collect_data_trainer(
        self,
        env,
        wandb_run=None,
        validate=False,
        save_img=False,
        log_name="collect",
        max_steps=None,
    ):
        return data_collect_api(
            cfg=self.cfg,
            env=env,
            wandb_run=wandb_run,
            save_img=save_img,
            log_name=log_name,
            max_steps=max_steps,
        )

    def del_env_data_file(self):
        data_path = getattr(self.cfg.env.collect, "data_save_path", None)
        if not data_path:
            return

        p = Path(data_path)
        if p.exists() and p.is_file():
            os.remove(p)
            print(f"[Support] Removed existing dataset: {p}")
