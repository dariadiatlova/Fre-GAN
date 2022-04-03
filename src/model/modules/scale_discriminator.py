import torch
import torch.nn as nn

from src.model.dwt import DiscreteWaveletTransform


class SubDiscriminator(nn.Module):
    def __init__(self, device: str, negative_slope: float):
        super(SubDiscriminator, self).__init__()
        self.DWT = DiscreteWaveletTransform(device)

        self.dwt_conv_layers = nn.ModuleList([
            nn.Conv1d(2, 128, kernel_size=(15,), stride=(1,), padding=7),
            nn.Conv1d(4, 128, kernel_size=(41,), stride=(2,), padding=20),
            ])

        self.convolution_layers = nn.ModuleList([
            nn.Conv1d(1, 128, kernel_size=(15,), stride=(1,), padding=7),
            nn.Conv1d(128, 128, kernel_size=(41,), groups=4, stride=(1,), padding=20),
            nn.Conv1d(128, 256, kernel_size=(41,), groups=16, stride=(1,), padding=20),
            nn.Conv1d(256, 512, kernel_size=(41,), groups=16, stride=(1,), padding=20),
            nn.Conv1d(512, 1024, kernel_size=(41,), groups=16, stride=(1,), padding=20),
            nn.Conv1d(1024, 1024, kernel_size=(41,), groups=16, stride=(1,), padding=20),
            nn.Conv1d(1024, 1024, kernel_size=(5,), stride=(1,), padding=2),
        ])

        self.convolution_out_layer = nn.Conv1d(1024, 1, kernel_size=(3,), stride=(1,), padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dwt_cache = None

    def __get_dwt(self, x):
        if self.dwt_cache is None:
            x_wavelet_high, x_wavelet_low = self.DWT(x)
            self.dwt_cache = [x_wavelet_high, x_wavelet_low]
            x = torch.cat(self.dwt_cache, dim=1)

        else:
            x_wavelet_high, x_wavelet_low = self.dwt_cache
            x_wavelet_high_one, x_wavelet_low_one = self.DWT(x_wavelet_high)
            x_wavelet_high_two, x_wavelet_low_two = self.DWT(x_wavelet_low)
            self.dwt_cache = [x_wavelet_high_one, x_wavelet_low_one, x_wavelet_high_two, x_wavelet_low_two]
            x = torch.cat(self.dwt_cache, dim=1)
            self.dwt_cache = None
        return x

    def forward(self, x: torch.Tensor):
        feature_match = []
        cached_x_outs = []
        _batch_size, _channels, _time_length = x.shape
        x_out = x.clone()

        for dwt_conv_layer in self.dwt_conv_layers:
            x_out = self.__get_dwt(x_out)
            x_out = dwt_conv_layer(x_out)
            cached_x_outs.append(x_out)

        for i, conv_layer in enumerate(self.convolution_layers):
            x = conv_layer(x)
            x = self.leaky_relu(x)
            feature_match.append(x)
            if i < 2:
                x = torch.cat([x, cached_x_outs[i]], dim=2)

        x = self.convolution_out_layer(x)
        feature_match.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_match


class RSD(nn.Module):
    """
    Consists of three sub-discriminators operating on different input scales: raw audio, 2 × down-sampled audio,
    and 4 × down-sampled audio.
    """
    def __init__(self, device, negative_slope):
        super(RSD, self).__init__()
        self.DWT = DiscreteWaveletTransform(device)
        self.dwt_conv_layers = nn.ModuleList([
            nn.Conv1d(2, 1, kernel_size=(1,), stride=(1,), padding=0),
            nn.Conv1d(4, 1, kernel_size=(1,), stride=(1,), padding=0),
            ])
        self.discriminators = nn.ModuleList([
            SubDiscriminator(device, negative_slope),
            SubDiscriminator(device, negative_slope),
            SubDiscriminator(device, negative_slope)
        ])
        self.dwt_cache = {}

    def __get_dwt(self, x, i, cache_name):
        if i == 0:
            x_wavelet_high, x_wavelet_low = self.DWT(x)
            self.dwt_cache[cache_name] = [x_wavelet_high, x_wavelet_low]
            x = torch.cat(self.dwt_cache[cache_name], dim=1)
        else:
            x_wavelet_high, x_wavelet_low = self.dwt_cache[cache_name]
            x_wavelet_high_one, x_wavelet_low_one = self.DWT(x_wavelet_high)
            x_wavelet_high_two, x_wavelet_low_two = self.DWT(x_wavelet_low)
            self.dwt_cache[cache_name] = [x_wavelet_high_one, x_wavelet_low_one, x_wavelet_high_two, x_wavelet_low_two]
            x = torch.cat(self.dwt_cache[cache_name], dim=1)
        return x

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_disc_real = []
        y_disc_generated = []
        real_feature_matches = []
        generated_feature_matches = []

        out = y.clone()
        out_hat = y_hat.clone()

        for i, conv_layer in enumerate(self.dwt_conv_layers):
            out = self.__get_dwt(out, i, "y_true")
            out = conv_layer(out)
            y_disc_real.append(out)

            out_hat = self.__get_dwt(out_hat, i, "y_gen")
            out_hat = conv_layer(out_hat)
            y_disc_generated.append(out_hat)

        for i, discriminator in enumerate(self.discriminators):
            if i < 2:
                y = y_disc_real[i]
                y_hat = y_disc_generated[i]

            y_real, real_feature_map = discriminator(y)
            y_gen, generated_feature_map = discriminator(y_hat)

            y_disc_real.append(y_real)
            real_feature_matches.append(real_feature_map)
            y_disc_generated.append(y_gen)
            generated_feature_matches.append(generated_feature_map)

        return y_disc_real, y_disc_generated, real_feature_matches, generated_feature_matches
