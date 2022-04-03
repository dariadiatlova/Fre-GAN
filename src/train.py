import datetime
from typing import Dict

import pytorch_lightning as pl
from commode_utils.callbacks import ModelCheckpointWithUploadCallback, PrintEpochResultCallback
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataloaders
from src.model.lightning_model import FreGan


def main(config: Dict):
    train_config = config["fre-gan"]
    wandb_config = config["wandb"]

    seed_everything(train_config["seed"])

    wandb_logger = WandbLogger(
        project=wandb_config["project"],
        log_model=False,
        offline=wandb_config["offline"],
        config=OmegaConf.to_container(config),
    )

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpointWithUploadCallback(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-val_loss={val/loss:.4f}",
        monitor="val/loss",
        every_n_epochs=train_config["save_every_epoch"],
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    progress_bar = TQDMProgressBar(refresh_rate=wandb_config["progress_bar_refresh_rate"])
    trainer = Trainer(
        max_epochs=train_config["epochs"],
        deterministic=True,
        check_val_every_n_epoch=train_config["val_every_epoch"],
        log_every_n_steps=wandb_config["log_every_n_steps"],
        logger=wandb_logger,
        gpus=train_config["n_gpus"],
        callbacks=[checkpoint_callback, progress_bar],
        resume_from_checkpoint=config.get("checkpoint", None),
    )

    train_loader, val_loader = get_dataloaders(config["dataset"])
    model = FreGan(config)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    config = OmegaConf.load("src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    main(config)
