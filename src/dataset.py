import librosa
import torch

from torch.utils.data import Dataset, DataLoader
from librosa.util import normalize

from collections import defaultdict
from typing import Dict, Optional, Tuple

from data import DATA_PATH
from src.model.melspectrogram import mel_spectrogram
from src.utils import get_file_names, load_audio, pad_input_audio_signal


class MelDataset(Dataset):
    """
    Class creates a dataloader for training and validation Fre-gan.
    It takes parameters from config.yaml and path to the folder with audios as a parameter.
    Please, look into config.yaml dataset section to find out about parameters in detail.
    """
    _log_file = "bad_samples.txt"

    def __init__(self, config: Dict, train: bool):
        for key, value in config.items():
            setattr(self, key, value)

        tsv_filepath = self.dataset_path
        self.audio_files = get_file_names(f"{DATA_PATH}/{tsv_filepath}", f"{DATA_PATH}/audio/", self.speaker_id)

        if train:
            self.audio_files = self.audio_files[:int(len(self.audio_files) * self.train_set)]
        else:
            self.audio_files = self.audio_files[int(len(self.audio_files) * self.train_set):]

        self._mel_cached = defaultdict()
        self._n_samples = len(self.audio_files)

    def __compute_mel_spectrogram(self, audio_file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function takes the path to an audio file and returns it's mel spectrogram.
        :param audio_file_path:
        :return: 2 tensors: mel-spectrogram and original audio padded/truncated to the target_audio_length.
        """
        audio = load_audio(audio_file_path, self.sr, self._log_file)
        resampled_audio = librosa.resample(audio, orig_sr=self.sr, target_sr=self.target_sr)

        # control amplitudes are in a range [-1, 1]
        audio = normalize(resampled_audio)
        audio = torch.FloatTensor(audio).type(torch.FloatTensor).unsqueeze(0)

        # pad from both sides or / truncate from right
        padded_audio = pad_input_audio_signal(audio, self.target_audio_length)

        spectrogram = mel_spectrogram(
            padded_audio, self.n_fft, self.n_mels, self.target_sr, self.hop_size, self.win_size, self.f_min, self.f_max
        )
        return spectrogram.squeeze(), padded_audio.squeeze(0)

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
        if self.dataset_size:
            return self.dataset_size
        return self._n_samples

    def __getitem__(self, idx: int) -> Optional[torch.Tensor]:
        item = self._mel_cached.get(idx, "empty")
        if item == "empty":
            return self.__cache_mel(idx)
        else:
            return item


def get_dataloaders(dataset_config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(MelDataset(dataset_config, train=True),
                                  batch_size=dataset_config["batch_size"],
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=dataset_config["num_workers"])

    val_dataloader = DataLoader(MelDataset(dataset_config, train=False),
                                batch_size=dataset_config["batch_size"],
                                shuffle=False,
                                pin_memory=True,
                                num_workers=dataset_config["num_workers"])

    # create dataloader for sampling 4, 5-seconds Audio in the end of each epoch and try to generate them
    test_config = dataset_config
    test_config["target_audio_length"] = test_config["target_sr"] * test_config["test_sample_size"]
    test_dataloader = DataLoader(MelDataset(dataset_config, train=False),
                                 batch_size=test_config["batch_size"],
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=dataset_config["num_workers"])

    return train_dataloader, val_dataloader, test_dataloader
