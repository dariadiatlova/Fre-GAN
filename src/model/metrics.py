import librosa
import numpy as np


from librosa.feature import melspectrogram, mfcc

from src.utils import load_audio


def mel_cepstral_distance(audio1: np.ndarray, audio2: np.ndarray, sr: int = 22050, n_mfcc: int = 20,
                          n_fft: int = 1024, hop_len: int = 256, win_len: int = 1024) -> float:
    """
    Implementation of  "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek:
    https://ieeexplore.ieee.org/document/407206.
    Returns mel-cepstral distance of two audio signals.
    """

    k = 10 / np.log(10) * np.sqrt(2)

    if np.max(abs(audio1)) > 1:
        audio1 /= np.max(abs(audio1))

    if np.max(abs(audio2)) > 1:
        audio2 /= np.max(abs(audio2))

    mel1 = melspectrogram(y=audio1, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len, power=2)
    mel2 = melspectrogram(y=audio2, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len, power=2)

    # the zeroth mfcc coefficient is omitted in the paper, and dct type 2 librosa uses while computing
    # mfcc is 2 times bigger than the one is used in the paper, so:
    mfcc1 = 1 / 2 * mfcc(y=audio1, sr=sr, S=np.log(mel1 + 1e-8), n_mfcc=n_mfcc)[1:]
    mfcc2 = 1 / 2 * mfcc(y=audio2, sr=sr, S=np.log(mel2 + 1e-8), n_mfcc=n_mfcc)[1:]

    mcd = k * np.mean(np.sqrt(np.sum((mfcc1 - mfcc2) ** 2, axis=1)))
    return mcd


def rmse_f0(audio1: np.ndarray, audio2: np.ndarray, sr: int = 22050, fmin: int = 0, fmax: int = 8000):
    pitches1 = librosa.core.piptrack(y=audio1, sr=sr, fmin=fmin, fmax=fmax)[0]
    pitches2 = librosa.core.piptrack(y=audio2, sr=sr, fmin=fmin, fmax=fmax)[0]
    return np.sqrt(np.mean((pitches1 - pitches2) ** 2))
