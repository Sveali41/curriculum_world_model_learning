import os

import torch
ROOTPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(ROOTPATH)

from modelBased.world_model.AttentionWM import AttentionWorldModel
from modelBased.data.datamodule import WMRLDataModule, WMRLDataset, WMRLDataset
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


# === Define Custom Datamodule for Validation ===
# This allows using 100% data for validation without modifying the original library code.
class ValidationDataModule(WMRLDataModule):
    def setup(self, stage=None):
        if self.direct_data is not None:
            loaded = self.direct_data
        else:
            loaded = np.load(self.data_dir, allow_pickle=True)
        
        # Create dataset
        data = WMRLDataset(loaded, self.hparams, self.replay_data)
        
        # Use 100% data for test
        # Create a Subset that covers the full range
        self.data_test = torch.utils.data.Subset(data, range(0, len(data)))
        # Train set is empty or just dummy
        self.data_train = torch.utils.data.Subset(data, range(0, 0))
        
        print(f"[ValidationDataModule] Used 100% data ({len(self.data_test)} samples) for validation.")

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            drop_last=False, # Allow small batches
            num_workers=self.hparams.n_cpu,
            pin_memory=True,
            persistent_workers=False
        )

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
    should_use_validation_mode = cfg.attention_model.freeze_weight
    
    if should_use_validation_mode:
        # User requested "Validation Only" mode (100% data, no split check)
        datamodule = ValidationDataModule(hparams=cfg.attention_model, replay_data=None)
    elif cfg.attention_model.continue_learning:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=replay_data)
    else:
        datamodule = WMRLDataModule(hparams=cfg.attention_model, replay_data=None)

    # logger
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="Local_Attention_Training", log_model=True, reinit=True)
        # Fix: Do NOT watch model during validation loop. 
        # Watching adds hooks to the persistent 'net', which reference this temporary 'wandb_logger'.
        # When 'wandb_logger' is GC'ed after this function returns, the hooks break.
        if not cfg.attention_model.freeze_weight:
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

    # ... (in run)
        # 保存 checkpoint
        model_pth = cfg.attention_model.model_save_path
        trainer.save_checkpoint(model_pth)
        if use_wandb:
            wandb.save(str(model_pth))
            wandb.save(model_pth)

        result["mode"] = "train"
        result["old_params"] = old_params
        result["fisher"] = fisher
        
        # Capture best validation loss
        best_score = trainer.checkpoint_callback.best_model_score
        result["best_loss"] = best_score.item() if best_score is not None else 0.0
        
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
        # For validation mode, return avg_val_loss as the loss metric
        return result["avg_val_loss"], None, result["net"]



if __name__ == "__main__":
    train()
