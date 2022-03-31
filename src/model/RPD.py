import torch
import torch.nn as nn
import torch.nn.functional as F

from DWT import DiscreteWaveletTransform


class SubDiscriminator(nn.Module):
    def __init__(self, period: int, negative_slope: float = 0.1):
        super(SubDiscriminator, self).__init__()
        self.period = period
        self.DWT = DiscreteWaveletTransform()

        self.dwt_conv_layers = nn.ModuleList([
            nn.Conv1d(2, 1, kernel_size=(1,)),
            nn.Conv1d(4, 1, kernel_size=(1,)),
            nn.Conv1d(8, 1, kernel_size=(1,))
        ])

        self.projection_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(1, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(1, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
        ])

        self.convolution_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=(1,), padding=(2, 0)),
        ])

        self.convolution_out_layer = nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1,), padding=(1, 0))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dwt_cache = None

    def __get_dwt(self, x):
        if self.dwt_cache is None:
            x_wavelet_high, x_wavelet_low = self.DWT(x)
            self.dwt_cache = [x_wavelet_high, x_wavelet_low]
            x = torch.cat(self.dwt_cache, dim=1)

        elif len(self.dwt_cache) == 2:
            x_wavelet_high, x_wavelet_low = self.dwt_cache
            x_wavelet_high_one, x_wavelet_low_one = self.DWT(x_wavelet_high)
            x_wavelet_high_two, x_wavelet_low_two = self.DWT(x_wavelet_low)
            self.dwt_cache = [x_wavelet_high_one, x_wavelet_low_one, x_wavelet_high_two, x_wavelet_low_two]
            x = torch.cat(self.dwt_cache, dim=1)

        else:
            x_wavelet_high_one, x_wavelet_low_one, x_wavelet_high_two, x_wavelet_low_two = self.dwt_cache

            x_wavelet_high_one, x_wavelet_low_one = self.DWT(x_wavelet_high_one)
            x_wavelet_high_two, x_wavelet_low_two = self.DWT(x_wavelet_low_one)
            x_wavelet_high_three, x_wavelet_low_three = self.DWT(x_wavelet_high_two)
            x_wavelet_high_four, x_wavelet_low_four = self.DWT(x_wavelet_low_two)

            self.dwt_cache = [x_wavelet_high_one, x_wavelet_low_one, x_wavelet_high_two, x_wavelet_low_two,
                              x_wavelet_high_three, x_wavelet_low_three, x_wavelet_high_four, x_wavelet_low_four]

            x = torch.cat(self.dwt_cache, dim=1)
        return x

    def forward(self, x: torch.Tensor):
        feature_map = []
        cached_x_outs = []
        _batch_size, _channels, _time_length = x.shape
        x_out = x.clone()

        for conv_layer, projection_layer in zip(self.dwt_conv_layers, self.projection_layers):
            x_out = self.__get_dwt(x_out)
            x_out = conv_layer(x_out)
            batch_size, channels, time_length = x_out.shape
            n_pad = self.period - (time_length % self.period)
            x_out = F.pad(x_out, (0, n_pad), "reflect")
            time_length += n_pad
            x_out = x_out.view(batch_size, channels, time_length // self.period, self.period)
            x_out = self.projection_layer(x_out)
            cached_x_outs.append(x_out)

        n_pad = self.period - (_time_length % self.period)
        x = F.pad(x, (0, n_pad), "reflect")
        _time_length = _time_length + n_pad
        x = x.view(_batch_size, _channels, _time_length // self.period, self.period)

        for i, layer in enumerate(self.convolution_layers):
            x = layer(x)
            x = self.leaky_relu(x)
            feature_map.append(x)
            if i < 3:
                x = torch.cat([x, cached_x_outs[i]], dim=2)

        x = self.self.convolution_out_layer(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_map


class RPD(nn.Module):
    """
    Comprises five sub-discriminators each of which accepts specific periodic parts of input audio:
        | input_audio [T] -> (2d-convolution) –> output_audio [p, T/p], where p is a period: {2, 3, 5, 7, 11}.
    """
    def __init__(self):
        super(RPD, self).__init__()
        self.discriminators = nn.ModuleList([
            SubDiscriminator(2),
            SubDiscriminator(3),
            SubDiscriminator(5),
            SubDiscriminator(7),
            SubDiscriminator(11)
        ])

    def forward(self, y, y_hat):
        y_disc_real = []
        y_disc_generated = []
        real_feature_maps = []
        generated_feature_maps = []

        for discriminator in self.discriminators:
            y_real, real_feature_map = discriminator(y)
            y_gen, generated_feature_map = discriminator(y_hat)

            y_disc_real.append(y_real)
            real_feature_maps.append(real_feature_map)
            y_disc_generated.append(y_gen)
            generated_feature_maps.append(generated_feature_map)

        return y_disc_real, y_disc_generated, real_feature_maps, generated_feature_maps
