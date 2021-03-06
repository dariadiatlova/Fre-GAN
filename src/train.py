from typing import Dict

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataloaders
from src.model.lightning_model import FreGan


def main(config: Dict):
    train_config = config["fre-gan"]
    wandb_config = config["wandb"]

    seed_everything(train_config["seed"])

    wandb_logger = WandbLogger(
        save_dir=wandb_config["save_dir"],
        project=wandb_config["project"],
        log_model=False,
        offline=wandb_config["offline"],
        config=config,
    )

    # define model checkpoint callback
    callbacks = ModelCheckpoint(dirpath=wandb_logger.experiment.dir,
                                monitor="train/generator_total_loss",
                                save_top_k=-1,
                                every_n_epochs=train_config["save_every_epoch"])

    progress_bar = TQDMProgressBar(refresh_rate=wandb_config["progress_bar_refresh_rate"])

    trainer = Trainer(
        max_epochs=train_config["epochs"],
        check_val_every_n_epoch=train_config["val_every_epoch"],
        log_every_n_steps=wandb_config["log_every_n_steps"],
        logger=wandb_logger,
        gpus=train_config["n_gpus"],
        callbacks=[callbacks, progress_bar]
    )

    train_loader, val_loader, test_loader = get_dataloaders(config["dataset"])
    if wandb_config["checkpoint_directory"] is not None:
        model = FreGan.load_from_checkpoint(checkpoint_path=wandb_config["checkpoint_directory"], config=config)
    else:
        model = FreGan(config, val_loader=test_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    config = OmegaConf.load("src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)

    main(config)
