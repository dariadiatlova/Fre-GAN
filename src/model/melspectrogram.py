# REFERENCE: https://github.com/jik876/hifi-gan/blob/master/meldataset.py
import torch.nn.functional as F
import torch

from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, constant: int = 1, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * constant)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024,
                    fmin=0, fmax=8000, center=False):

    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec
