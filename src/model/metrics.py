import librosa
import numpy as np
import torch

from torchaudio.transforms import MFCC


def mel_cepstral_distance(audio1: torch.Tensor, audio2: torch.Tensor, sr: int = 22050, n_mfcc: int = 40,
                          device: str = "cuda") -> float:
    """
    Implementation of  "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek:
    https://ieeexplore.ieee.org/document/407206.
    Returns mel-cepstral distance of two audio signals.
    """

    k = 10 / np.log(10) * np.sqrt(2)

    if torch.max(torch.abs(audio1)) > 1:
        audio1 /= torch.max(torch.abs(audio1))

    if torch.max(torch.abs(audio2)) > 1:
        audio2 /= torch.max(torch.abs(audio2))

    mfcc = MFCC(sample_rate=sr, n_mfcc=n_mfcc, log_mels=True).to(device)

    # the zeroth mfcc coefficient is omitted in the paper, and dct type 2 librosa uses while computing
    # mfcc is 2 times bigger than the one is used in the paper, so:
    mfcc1 = 1 / 2 * mfcc(audio1)[1:]
    mfcc2 = 1 / 2 * mfcc(audio2)[1:]

    mcd = k * torch.mean(torch.sqrt(torch.sum((mfcc1 - mfcc2) ** 2, dim=1)))
    return mcd


def rmse_f0(audio1: np.ndarray, audio2: np.ndarray, sr: int = 22050, fmin: int = 0, fmax: int = 8000):
    pitches1 = librosa.core.piptrack(y=audio1, sr=sr, fmin=fmin, fmax=fmax)[0]
    pitches2 = librosa.core.piptrack(y=audio2, sr=sr, fmin=fmin, fmax=fmax)[0]
    return np.sqrt(np.mean((pitches1 - pitches2) ** 2))
