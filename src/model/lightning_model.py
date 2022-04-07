from typing import Dict
from itertools import chain

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule

from src.model.losses import generator_loss, discriminator_loss
from src.model.modules.generator import RCG
from src.model.modules.period_discriminator import RPD
from src.model.modules.scale_discriminator import RSD
from src.model.metrics import mel_cepstral_distance, rmse_f0


class FreGan(LightningModule):
    def __init__(self, config: Dict, inference: bool = False):
        super().__init__()
        self.save_hyperparameters()
        fre_gan_config = config["fre-gan"]

        for key, value in fre_gan_config.items():
            setattr(self, key, value)

        self.generator = RCG(config["rcg"])
        if inference:
            self.generator.remove_weight_norm()

        self.rp_discriminator = RPD(self.current_device, config["rcg"]["negative_slope"])
        self.sp_discriminator = RSD(self.current_device, config["rcg"]["negative_slope"])

        self.generator_loss_function = generator_loss
        self.discriminator_loss_function = discriminator_loss

    def _generator_shared_step(self, y_gen, y_true, step):
        y_rpd_real, y_rpd_gen, real_fm_rpd, gen_fm_rpd = self.rp_discriminator(y_true.unsqueeze(1), y_gen)
        y_rsd_real, y_rsd_gen, real_fm_rsd, gen_fm_rsd = self.sp_discriminator(y_true.unsqueeze(1), y_gen)

        total_gen_loss, adv_loss, fm_loss, stft_loss = self.generator_loss_function(
            y_rpd_gen, y_rsd_gen, y_true, y_gen, real_fm_rpd, gen_fm_rpd, real_fm_rsd, gen_fm_rsd, self.current_device
        )

        gen_log_dict = {f"{step}/generator_total_loss": total_gen_loss,
                        f"{step}/generator_adversarial_loss": adv_loss,
                        f"{step}/feature_matching_loss": fm_loss,
                        f"{step}/stft_loss": stft_loss}

        self.log_dict(gen_log_dict, on_step=True, on_epoch=False)
        return total_gen_loss

    def _discriminator_shared_step(self, y_gen, y_true, step):
        y_rpd_real, y_rpd_gen, real_fm_rpd, gen_fm_rpd = self.rp_discriminator(y_true.unsqueeze(1), y_gen.detach())
        y_rsd_real, y_rsd_gen, real_fm_rsd, gen_fm_rsd = self.sp_discriminator(y_true.unsqueeze(1), y_gen.detach())

        rpd_loss, rsd_loss = self.discriminator_loss_function(y_rpd_real, y_rpd_gen, y_rsd_real, y_rsd_gen)
        total_discriminator_loss = rpd_loss + rsd_loss
        disc_log_dict = {f"{step}/discriminator_total_loss": total_discriminator_loss,
                         f"{step}/period_discriminator_loss": rpd_loss,
                         f"{step}/scale_discriminator_loss": rsd_loss}

        self.log_dict(disc_log_dict, on_step=True, on_epoch=False)

        return total_discriminator_loss

    def _compute_metrics(self, batch):
        mel_spectrogram, y_true = batch
        ys_gen = self.generator(mel_spectrogram).squeeze(1).detach()
        ys_true = y_true.detach()

        rmses = []
        mcds = []

        for y_true, y_gen in zip(ys_true, ys_gen):
            mcds.append(mel_cepstral_distance(y_true, y_gen))
            rmses.append(rmse_f0(y_true.cpu().numpy(), y_gen.detach().cpu().numpy()))
        return torch.mean(torch.tensor(mcds)), np.mean(rmses)

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            opt_d = torch.optim.Adam(chain(self.rp_discriminator.parameters(), self.sp_discriminator.parameters()),
                                     lr=self.lr * 0.1, betas=(self.b1, self.b2))
        else:
            opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            opt_d = torch.optim.AdamW(chain(self.rp_discriminator.parameters(), self.sp_discriminator.parameters()),
                                      lr=self.lr * 0.1, betas=(self.b1, self.b2))

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.lr_decay)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def forward(self, mel_spectrogram):
        y_gen = self.generator(mel_spectrogram)
        return y_gen

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.generator.train()
        self.rp_discriminator.train()
        self.sp_discriminator.train()

        mel_spectrogram, y_true = batch

        if optimizer_idx == 0:
            y_gen = self.generator(mel_spectrogram)
            return self._generator_shared_step(y_gen, y_true, "train")

        # train / val discriminators
        if optimizer_idx == 1:
            y_gen = self.generator(mel_spectrogram)
            return self._discriminator_shared_step(y_gen, y_true, "train")

    def validation_step(self, batch, batch_idx):
        self.generator.eval()
        self.rp_discriminator.eval()
        self.sp_discriminator.eval()

        log_dict = {}
        with torch.no_grad():
            rmse, mcds = self._compute_metrics(batch)
            log_dict["val/RMSE_f0"] = rmse
            log_dict["val/MCD"] = mcds

            self.log_dict(log_dict, on_step=True, on_epoch=False)

            # log val generator loss
            mel_spectrogram, y_true = batch
            y_gen = self.generator(mel_spectrogram)
            self._generator_shared_step(y_gen, y_true, "val")

            # log val discriminator loss
            mel_spectrogram, y_true = batch
            y_gen = self.generator(mel_spectrogram)
            self._discriminator_shared_step(y_gen, y_true, "val")

    def on_train_epoch_end(self):
        if self.current_epoch % self.save_every_epoch == 0:
            self.generator.eval()

            with torch.no_grad():
                for batch in self.val_dataloader():
                    mels, wavs = batch
                    generated_samples = self.generator(mels)
                    # grab only 1 batch
                    break

                for i, (original, generated) in enumerate(zip(wavs, generated_samples)):
                    generated = generated.squeeze(0).squeeze(0).detach().cpu().numpy()
                    original = original.detach().cpu().numpy()

                    self.logger.experiment.log(
                        {"generated_audios": wandb.Audio(generated, caption=f"Generated_{i}", sample_rate=22050)}
                    )

                    self.logger.experiment.log(
                        {"original_audios": wandb.Audio(original, caption=f"Original_{i}", sample_rate=22050)}
                    )
