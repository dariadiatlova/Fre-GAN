import torch

from typing import Dict, Optional
from collections import defaultdict

from torch.utils.data import Dataset

from data import DATA_PATH
from src.dataset.utils import get_file_names, load_audio, pad_input_audio_signal, normalize_amplitudes, \
    get_mel_spectrogram


class MelDataset(Dataset):
    """
    Class creates a dataloader for training and validation Fre-gan.
    It takes parameters from config.yaml and path to the folder with audios as a parameter.
    Please, look into config.yaml dataset section to find out about parameters in detail.
    """
    _log_file = "bad_samples.log"

    def __init__(self, config: Dict, train: bool):
        for key, value in config.items():
            setattr(self, key, value)

        if train:
            tsv_filepath = self.train_filepath
        else:
            tsv_filepath = self.val_filepath

        self.audio_files = get_file_names(f"{DATA_PATH}/{tsv_filepath}", f"{DATA_PATH}/audio/")
        self._mel_cached = defaultdict(lambda: None)

        self._target_audio_size = self.sr * self.target_audio_length
        self._n_samples = len(self.audio_files)

    def __compute_mel_spectrogram(self, audio_file_path: str) -> torch.Tensor:
        """
        Function takes the path to an audio file and returns it's mel spectrogram.
        :param audio_file_path:
        :return: torch.Tensor
        """
        audio = load_audio(audio_file_path, self.sr, self._log_file)

        # check that input channel matches config
        assert len(audio.shape) == self.n_channels, f"Audio file {audio_file_path} have different number of channels!" \
                                                    f"Expected input audio to have {self.n_channels}, " \
                                                    f"got {audio.shape[0]} instead."

        # control amplitudes are in a range [-1, 1]
        norm_audio = normalize_amplitudes(audio)

        # pad from both sides or / truncate from right
        padded_audio = pad_input_audio_signal(norm_audio, self._target_audio_size)
        assert padded_audio.shape[0] == self._target_audio_size, f"Expected all audios to be {self._target_audio_size}" \
                                                                 f"length, but got {audio_file_path} of " \
                                                                 f"length {padded_audio.shape[0]}"

        mel_spectrogram_db = get_mel_spectrogram(padded_audio, self.hop_size, self.n_mels, self.n_fft,
                                                 self.sr, self.f_max, self.normalize_spec)
        return mel_spectrogram_db

    def __cache_mel(self, idx: int) -> torch.Tensor:
        """
        Reading .wav file from disk and converting them to mel takes some time, so we'll cache the values to
        reuse mels over one training iteration.
        :param idx:
        :return:
        """
        mel_spec = self.__compute_mel_spectrogram(self.audio_files[idx])
        self._mel_cached[idx] = mel_spec
        return mel_spec

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        if self._mel_cached[idx]:
            return self._mel_cached[idx]
        else:
            return self.__cache_mel(idx)