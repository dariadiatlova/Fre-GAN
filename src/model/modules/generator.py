import torch
import torch.nn as nn

from typing import Dict

from torch.nn.utils import weight_norm, remove_weight_norm

from src.utils import init_weights


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, negative_slope: int):
        super(DilatedResidualBlock, self).__init__()
        self.dilated_convolutions = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(1,),
                                  padding=self.__get_padding_size(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(3,),
                                  padding=self.__get_padding_size(kernel_size, 3))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(5,),
                                  padding=self.__get_padding_size(kernel_size, 5))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(7,),
                                  padding=self.__get_padding_size(kernel_size, 7))),
        ])
        self.dilated_convolutions.apply(init_weights)

        self.convolutions = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(1,),
                                  padding=self.__get_padding_size(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(1,),
                                  padding=self.__get_padding_size(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(1,),
                                  padding=self.__get_padding_size(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size=(kernel_size,), stride=(1,), dilation=(1,),
                                  padding=self.__get_padding_size(kernel_size, 1))),

        ])
        self.convolutions.apply(init_weights)

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    @staticmethod
    def __get_padding_size(kernel_size: int, dilation: int):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x: torch.Tensor) -> None:
        for dilated_conv_layer, conv_layer in zip(self.dilated_convolutions, self.convolutions):
            residual = dilated_conv_layer(self.leaky_relu(x))
            residual = conv_layer(self.leaky_relu(residual))
            x = x + residual
        return x


class RCG(nn.Module):
    def __init__(self, config: Dict):
        super(RCG, self).__init__()

        for key, value in config.items():
            setattr(self, key, value)

        self.conditioning_level = self.n_conv_blocks - self.top_k

        self.conv1 = weight_norm(nn.Conv1d(80, 512, kernel_size=(7,), stride=(1,), padding=(3,)))
        self.conv_out = weight_norm(nn.Conv1d(16, 1, kernel_size=(7,), stride=(1,), padding=(3,)))
        self.conv_out.apply(init_weights)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

        self.conv_transposed_block = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=(16,), stride=(8,), padding=(4,))),
            weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=(8,), stride=(4,), padding=(2,))),
            weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=(4,), stride=(2,), padding=(1,))),
            weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))),
            weight_norm(nn.ConvTranspose1d(32, 16, kernel_size=(4,), stride=(2,), padding=(1,)))
        ])
        self.conv_transposed_block.apply(init_weights)

        self.residual_up_sampling = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          weight_norm(nn.Conv1d(128, 64, kernel_size=(1,)))),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          weight_norm(nn.Conv1d(64, 32, kernel_size=(1,)))),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          weight_norm(nn.Conv1d(32, 16, kernel_size=(1,))))
        ])
        self.residual_up_sampling.apply(init_weights)

        self.condition_up_sampling = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(80, 256, kernel_size=(16,), stride=(8,), padding=(4,))),
            weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=(8,), stride=(4,), padding=(2,))),
            weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=(4,), stride=(2,), padding=(1,))),
            weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))),
        ])
        self.condition_up_sampling.apply(init_weights)

        self.residual_blocks = self.__initialise_resblocks()

        self.res_block_output = None
        self.dilated_residual = None
        self.mel_spectrogram = None

    def __initialise_resblocks(self):
        """
        Creates multiple residual blocks for MRF block
        """
        resblocks = nn.ModuleList()
        for channels in self.channels:
            for kernel_size in self.kernel_sizes:
                resblocks.append(DilatedResidualBlock(channels, kernel_size, self.negative_slope))
        return resblocks

    def __conditioning(self, n_block: int, x: torch.Tensor):
        """
        Function return input conditioned on input mel spectrogram if needed to be.
        In the article conditioning is happening for last k=4 conv blocks.
        :param n_block: int, serial block number.
        :param x: torch.Tensor, input of the n_block.
        :return: torch.Tensor
        """

        if n_block < self.conditioning_level:
            return x

        # condition x on mel_spectrogram
        ups_idx = n_block - self.conditioning_level
        self.mel_spectrogram = self.condition_up_sampling[ups_idx](self.mel_spectrogram)
        x += self.mel_spectrogram

        self.__cache_output(n_block, x)
        return x

    def __cache_output(self, n_block: int, x: torch.Tensor) -> None:
        """
        MRF module returns the sum of outputs from multiple residual blocks,
        upsamples and store residual output in self.res_block_output variable
        """
        if n_block <= self.conditioning_level:
            return

        if self.res_block_output is not None:
            x = self.res_block_output
        self.res_block_output = self.residual_up_sampling[n_block - self.conditioning_level - 1](x)

    def __dilation(self, residual_idx, x):
        if self.dilated_residual is not None:
            self.dilated_residual += self.residual_blocks[residual_idx](x)
        else:
            self.dilated_residual = self.residual_blocks[residual_idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.res_block_output = None
        self.mel_spectrogram = x
        x = self.conv1(x)

        for n_block in range(self.n_conv_blocks):
            x = self.__conditioning(n_block, x)
            x = self.leaky_relu(x)
            x = self.conv_transposed_block[n_block](x)

            self.dilated_residual = None
            for k in range(self.num_kernels):
                residual_idx = n_block * self.num_kernels + k
                self.__dilation(residual_idx, x)

            x = self.dilated_residual / self.num_kernels
            if self.res_block_output is not None:
                self.res_block_output = self.res_block_output + x

        x = self.leaky_relu(self.res_block_output)
        x = self.conv_out(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv_out)
        for l in self.conv_transposed_block:
            remove_weight_norm(l)
        for l in self.condition_up_sampling:
            remove_weight_norm(l)
        for block in self.residual_blocks:
            for l1, l2 in zip(block.dilated_convolutions, block.convolutions):
                remove_weight_norm(l1)
                remove_weight_norm(l2)
        for l in self.residual_up_sampling:
            remove_weight_norm(l[1])
