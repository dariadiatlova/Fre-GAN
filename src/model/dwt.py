from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class DiscreteWaveletTransform(nn.Module):
    """
    The input audio signal is convolved by two filters: low-pass filter (g) and high-pass filter (h).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, stride: int = 2,
                 pad_type='reflect', wave_name='haar'):

        super(DiscreteWaveletTransform, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.wave_name = wave_name
        self.pad_type = pad_type

        self.kernel_size = None
        self.filter_low = None
        self.filter_high = None

        self.__get_filters()
        self.__initialize()

    def __get_filters(self):
        wavelet = pywt.Wavelet(self.wave_name)
        band_low = torch.tensor(wavelet.rec_lo).type(torch.FloatTensor)
        band_high = torch.tensor(wavelet.rec_hi).type(torch.FloatTensor)
        length_band = band_low.size()[0]

        self.filter_low = torch.zeros(length_band)
        self.filter_high = torch.zeros(length_band)
        self.filter_low[:None] = band_low
        self.filter_high[:None] = band_high

        self.kernel_size = length_band

    def __initialize(self):
        self.filter_low = self.filter_low[None, None, :]
        self.filter_high = self.filter_high[None, None, :]
        self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, 1, audio_length] -> ([bs, 1, audio_length // 2], [bs, 1, audio_length // 2])
        x = F.pad(x, pad=self.pad_sizes, mode=self.pad_type)
        g_filter = F.conv1d(x, self.filter_low, stride=self.stride)
        h_filter = F.conv1d(x, self.filter_high, stride=self.stride)
        return g_filter, h_filter
