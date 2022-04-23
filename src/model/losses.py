from typing import Dict

import torch
import torch.nn.functional as F

from src.dataset import get_mel_spectrogram


def discriminator_loss(period_d_outs_real, period_d_outs_gen, scale_d_outs_real, scale_d_outs_gen):
    """
    Implementation of least-squares GAN objective: E|y_true - 1|**2 + E|y_fake|**2,
    adopted for for 2 discriminators. Each variable is a list of 4 tensors of 2d / 3d tensors.

    :param period_d_outs_real: List[torch.Tensor], output of RPD for real audio
    :param period_d_outs_gen: List[torch.Tensor], output of RPD for generated audio
    :param scale_d_outs_real: List[torch.Tensor], output of RSD for real audio
    :param scale_d_outs_gen: List[torch.Tensor], output of RSD for generated audio
    :return: Tuple[int, int] loss for RPD, RSD
    """
    rpd_loss = 0
    rsd_loss = 0

    for pdlr, pdlg, sdlr, sdlg in zip(period_d_outs_real, period_d_outs_gen, scale_d_outs_real, scale_d_outs_gen):
        rpd_loss += torch.mean((pdlr - torch.ones_like(pdlr)) ** 2) + torch.mean(pdlg ** 2)
        rsd_loss += torch.mean((sdlr - torch.ones_like(sdlr)) ** 2) + torch.mean(sdlg ** 2)

    return rpd_loss, rsd_loss


def _mel_spectrogram_loss(y_true, y_gen, dataset_config: Dict):
    # print(y_true.shape, y_gen.shape)
    mel_true = get_mel_spectrogram(y_true, dataset_config["hop_size"], dataset_config["win_size"],
                                   dataset_config["n_mels"], dataset_config["n_fft"], dataset_config["power"],
                                   dataset_config["target_sr"], dataset_config["f_min"], dataset_config["f_max"])

    mel_gen = get_mel_spectrogram(y_gen, dataset_config["hop_size"], dataset_config["win_size"],
                                  dataset_config["n_mels"], dataset_config["n_fft"], dataset_config["power"],
                                  dataset_config["target_sr"], dataset_config["f_min"], dataset_config["f_max"])
    return F.l1_loss(mel_true, mel_gen)


def _feature_matching_loss(period_fm_real, period_fm_gen, scale_fm_real, scale_fm_gen):
    rpd_fm_loss = 0
    rsd_fm_loss = 0
    # iterate trough sub-discriminators outputs
    for pfmr, pfmg, sfmr, sfmg in zip(period_fm_real, period_fm_gen, scale_fm_real, scale_fm_gen):
        # iterate trough i'th discriminator layer output / dwt representation
        for i_pfmr, i_pfmg, i_sfmr, i_sfmg in zip(pfmr, pfmg, sfmr, sfmg):
            rpd_fm_loss += torch.mean(abs(i_pfmr - i_pfmg))
            rsd_fm_loss += torch.mean(abs(i_sfmr - i_sfmg))
    return rpd_fm_loss, rsd_fm_loss


def generator_loss(period_d_outs_gen, scale_d_outs_gen, y_true, y_gen,
                   period_fm_real, period_fm_gen, scale_fm_real, scale_fm_gen,
                   dataset_config):

    # parameters from the paper
    lambda_mel = 45
    lambda_fm = 2

    prd_adv_l = 0
    srd_adv_l = 0

    for prd, srd in zip(period_d_outs_gen, scale_d_outs_gen):
        prd_adv_l += torch.mean((prd - torch.ones_like(prd)) ** 2)
        srd_adv_l += torch.mean((srd - torch.ones_like(srd)) ** 2)

    rpd_fm_loss, rsd_fm_loss = _feature_matching_loss(period_fm_real, period_fm_gen, scale_fm_real, scale_fm_gen)
    stft_loss = _mel_spectrogram_loss(y_true, y_gen, dataset_config)

    total_generator_loss = prd_adv_l + srd_adv_l + (lambda_fm * (rpd_fm_loss + rsd_fm_loss)) + (lambda_mel * stft_loss)
    return total_generator_loss, prd_adv_l + srd_adv_l, lambda_fm * (rpd_fm_loss + rsd_fm_loss), stft_loss * lambda_mel
