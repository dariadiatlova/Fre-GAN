import random

import numpy as np
import pandas as pd
import re

import librosa
import torch

from typing import Optional

from torchaudio import transforms
import soundfile as sf


def write_wav_file(data: np.ndarray, filepath: str, sample_rate: int = 22050):
    if np.max(abs(data)) > 1:
        data /= max(abs(data))
    sf.write(filepath, data, sample_rate)


def get_file_names(tsv_filepath: str, root_path: str, speaker_id: Optional[int] = None):
    """
    Function takes the path the tsv file and returns the column with filenames, changed from .mp3 to .wav.
    :param tsv_filepath: str
    :param root_path: str path to the root directory with the extracted filenames
    :param speaker_id: int, str id of speaker to use for training and validation
    :return: ndarray
    """
    df = pd.read_csv(tsv_filepath, sep='\t')
    if speaker_id:
        # use only 1 speaker audios for training / validation
        df = df[df.client_id == speaker_id]
    mp3_filenames = np.array(df.path)
    pattern = re.compile(".*(?=.mp3)")
    wav_filenames = []
    for filename in mp3_filenames:
        if pattern.search(filename) is not None:
            wav_filenames.append(root_path + pattern.search(filename).group() + ".wav")
    return np.array(wav_filenames)


def load_audio(audio_filepath: str, sample_rate, logger_file: Optional[str] = None):
    """
    Function takes the filepath to the audio in .wav format and return a numpy array.
    :param audio_filepath: str, path the .wav file
    :param sample_rate: sample_rate used for reading the .wav file.
    :param logger_file: path to the txt file to write logs.
    :return:
    """
    try:
        return librosa.load(audio_filepath, sr=sample_rate)[0]
    except Exception as e:
        if logger_file:
            with open(logger_file, "w") as f:
                f.write(f"Couldn't load the file: {audio_filepath}.")
        else:
            raise e
        return


def pad_crop_audio(input_signal: np.ndarray, target_length: int) -> torch.Tensor:
    """
    Function takes input signal and align it with the target signal length.
    :param input_signal: np.ndarray
    :param target_length: int, seconds * sample_rate
    :return:
    """
    if len(input_signal) >= target_length:
        start_sample = random.randint(0, input_signal.shape[0] - target_length)
        input_signal = input_signal[start_sample:target_length + start_sample]
        return torch.from_numpy(input_signal).type(torch.FloatTensor)

    else:
        pad_size = (target_length - len(input_signal))
        padded_signal = np.concatenate([np.zeros(pad_size // 2), input_signal, np.zeros(pad_size - (pad_size // 2))])
        return torch.from_numpy(padded_signal).type(torch.FloatTensor)


def normalize_amplitudes(signal: np.ndarray) -> np.ndarray:
    max_value = max(abs(signal))
    if max_value > 1:
        signal = signal / max_value
    return signal


def get_mel_spectrogram(input_audio: torch.Tensor, hop_length: int, n_mels: int, n_fft: int, power: int,
                        sample_rate: int, f_min: int = 0, f_max: int = 8000, normalized: bool = True) -> torch.Tensor:
    mel_spectrogram = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                n_mels=n_mels, f_max=f_max, normalized=normalized,
                                                onesided=True)
    device = input_audio.device
    mel_spectrogram.power = power
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max).T
    mel_basis = torch.from_numpy(mel_basis)
    mel_spectrogram.mel_scale.fb.copy_(mel_basis)
    mel = mel_spectrogram(input_audio.to("cpu")).clamp_(min=1e-5).log_()
    return mel.to(device)
