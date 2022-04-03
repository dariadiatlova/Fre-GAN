from typing import Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule

from src.model.losses import generator_loss, discriminator_loss
from src.model.modules.generator import RCG
from src.model.modules.period_discriminator import RPD
from src.model.modules.scale_discriminator import RSD
from src.model.metrics import mel_cepstral_distance, rmse_f0


class FreGan(LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        fre_gan_config = config["fre-gan"]

        for key, value in fre_gan_config.items():
            setattr(self, key, value)

        self.generator = RCG(config["rcg"])
        self.rp_discriminator = RPD(self.device, config["rcg"]["negative_slope"])
        self.sp_discriminator = RSD(self.device, config["rcg"]["negative_slope"])

        self.generator_loss_function = generator_loss
        self.discriminator_loss_function = discriminator_loss

    def _shared_step(self, batch, optimizer_idx, step: str) -> Dict:
        mel_spectrogram, y_true = batch

        # train / val generator
        if optimizer_idx == 0:
            y_gen = self.generator(mel_spectrogram)

            y_rpd_real, y_rpd_gen, real_fm_rpd, gen_fm_rpd = self.rpd(y_true.unsqueeze(1), y_gen)
            y_rsd_real, y_rsd_gen, real_fm_rsd, gen_fm_rsd = self.rsd(y_true.unsqueeze(1), y_gen)

            total_gen_loss, adv_loss, fm_loss, stft_loss = self.generator_loss_function(
                y_rpd_gen, y_rsd_gen, y_true, y_gen, real_fm_rpd, gen_fm_rpd, real_fm_rsd, gen_fm_rsd, self.device
            )

            gen_log_dict = {f"{step}/generator_total_loss": total_gen_loss,
                            f"{step}/generator_adversarial_loss": adv_loss,
                            f"{step}/feature_matching_loss": fm_loss,
                            f"{step}/stft_loss": stft_loss}

            if step == "val":
                with torch.no_grad():
                    rmse, mcds = self._compute_metrics(batch)
                    gen_log_dict["rmse"] = rmse
                    gen_log_dict["RMSE_f0"] = rmse
                    gen_log_dict["MCD"] = mcds

            self.log_dict(gen_log_dict, on_step=True, on_epoch=False)
            return total_gen_loss

        # train / val discriminators
        if optimizer_idx == 1:
            y_gen = self.generator(mel_spectrogram)

            y_rpd_real, y_rpd_gen, real_fm_rpd, gen_fm_rpd = self.rpd(y_true.unsqueeze(1), y_gen.detach())
            y_rsd_real, y_rsd_gen, real_fm_rsd, gen_fm_rsd = self.rsd(y_true.unsqueeze(1), y_gen.detach())

            rpd_loss, rsd_loss = self.discriminator_loss_function(y_rpd_real, y_rpd_gen, y_rsd_real, y_rsd_gen)
            total_discriminator_loss = rpd_loss + rsd_loss
            disc_log_dict = {f"{step}/discriminator_total_loss": total_discriminator_loss,
                             f"{step}/period_discriminator_loss": rpd_loss,
                             f"{step}/scale_discriminator_loss": rsd_loss}

            self.log_dict(disc_log_dict, on_step=True, on_epoch=False)

            return total_discriminator_loss

    def _compute_metrics(self, batch):
        mel_spectrogram, y_true = batch
        ys_gen = self.generator(y_true).squeese(1).deatch().cpu().numpy()
        ys_true = y_true.detach().cpu().numpy()

        rmses = []
        mcds = []

        for y_true, y_gen in zip(ys_true, ys_gen):
            mcds.append(mel_cepstral_distance(y_true, y_gen))
            rmses.append(rmse_f0(y_true, y_gen))
        return np.mean(mcds), np.mean(rmses)

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        else:
            opt_g = torch.optim.AdamW(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def forward(self, mel_spectrogram):
        y_gen = self.generator(mel_spectrogram)
        return y_gen

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.generator.tran()
        self.rp_discriminator.train()
        self.sp_discriminator.train()

        return self._shared_step(batch, optimizer_idx, "train")

    def validation_step(self, batch, batch_idx, optimizer_idx):
        self.generator.eval()
        self.rp_discriminator.eval()
        self.sp_discriminator.eval()
        loss = self._shared_step(batch, batch_idx, "val")
        return loss
