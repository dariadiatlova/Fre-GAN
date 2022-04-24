import random
import re

from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F


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


def pad_input_audio_signal(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Function takes input signal and align it with the target signal length.
    :param audio: np.ndarray
    :param target_length: int, seconds * sample_rate
    :return:
    """
    if audio.size(1) >= target_length:
        max_audio_start = audio.size(1) - target_length
        audio_start = random.randint(0, max_audio_start)
        audio = audio[:, audio_start:audio_start + target_length]
    else:
        audio = F.pad(audio, (0, target_length - audio.size(1)), 'constant')
    return audio


def normalize_amplitudes(signal: np.ndarray) -> np.ndarray:
    max_value = max(abs(np.max(signal)), abs(np.min(signal)))
    if max_value > 1:
        signal = signal / max_value
    return signal


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
