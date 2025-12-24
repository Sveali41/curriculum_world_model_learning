import os

import torch
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(ROOTPATH)

from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.data.datamodule import WMRLDataModule
from modelBased.common.utils import PROJECT_ROOT, get_env
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import numpy as np
from modelBased.common.utils import TRAINER_PATH


@hydra.main(version_base=None, config_path=str(TRAINER_PATH / "conf"), config_name="config_test")
def train(cfg: DictConfig):
    run(cfg)

def compare_params(net, old_params):
    if old_params is None:
        print('old params is None, skip comparison')
        return
    print("------ Comparing old_params to current model params ------")
    for name, param in net.named_parameters():
        if name in old_params:
            diff = (param.detach().cpu() - old_params[name]).abs().max().item()
            print(f"{name:40s} diff = {diff:.8f}")


def run(
    cfg: DictConfig,
    net: AttentionWorldModel,
    old_params=None,
    fisher=None,
    layout=None,
    replay_data=None
):
    print(f'*************************Data set: {cfg.attention_model.data_dir}************************')

    use_wandb = cfg.attention_model.use_wandb
    fisher_beta = float(getattr(cfg.attention_model, "fisher_beta", 0.5))

    # datamodule
    if cfg.attention_model.continue_learning:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=replay_data)
    else:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=None)

    # logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local_Attention_Training", log_model=True, reinit=True)
        wandb_logger.experiment.watch(net, log='all', log_freq=1000)

    # callbacks
    metric_to_monitor = 'avg_val_loss_wm'
    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor,
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric_to_monitor,
        mode="min",
        dirpath=os.path.dirname(cfg.attention_model.model_save_path),
        filename="att-{epoch:02d}-{avg_val_loss_wm:.5f}",
        verbose=True
    )

    # trainer
    trainer = pl.Trainer(
        precision=32,
        logger=wandb_logger if use_wandb else None,
        max_epochs=cfg.attention_model.n_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        deterministic=False,
    )

    result = {
        "mode": None,            
        "net": net,              
        "old_params": None,
        "fisher": None,
        "avg_val_loss": None,
    }

    # consolidation
    net.set_consolidation(old_params, fisher, load_weights=False)

    if cfg.attention_model.freeze_weight:
        # ===== validation =====
        avg_val_loss = trainer.validate(net, datamodule)

        result["mode"] = "val"
        result["avg_val_loss"] = avg_val_loss
        return result

    else:
        # ===== training =====
        trainer.fit(net, datamodule)

        # 保存旧参数
        old_params = net.save_old_params()

        # 计算 Fisher
        fisher_samples = int(getattr(cfg.attention_model, "fisher_samples", 3000))
        scale_factor = cfg.attention_model.scale_factor
        new_fisher = net.compute_fisher(
            datamodule.train_dataloader(),
            samples=fisher_samples,
            scale_factor=scale_factor
        )

        # EMA 合并 Fisher
        if fisher is not None:
            fisher = {
                k: (1.0 - fisher_beta) * fisher[k] + fisher_beta * new_fisher[k]
                for k in new_fisher
            }
        else:
            fisher = new_fisher

        # 保存 checkpoint
        model_pth = cfg.attention_model.model_save_path
        trainer.save_checkpoint(model_pth)
        if use_wandb:
            wandb.save(str(model_pth))
            wandb.save(model_pth)

        result["mode"] = "train"
        result["old_params"] = old_params
        result["fisher"] = fisher
        return result


def train_api(
    cfg: DictConfig,
    net: AttentionWorldModel,
    old_params=None,
    fisher=None,
    env_layout=None,
    replay_data=None
):
    result = run(
        cfg,
        net,
        old_params=old_params,
        fisher=fisher,
        layout=env_layout,
        replay_data=replay_data
    )

    if result["mode"] == "train":
        return result["old_params"], result["fisher"], result["net"]

    else:
        return result["avg_val_loss"], None, result["net"]



if __name__ == "__main__":
    train()
