import librosa
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple

from data import DATA_PATH
from src.utils import get_file_names, load_audio, get_mel_spectrogram, pad_crop_audio


class MelDataset(Dataset):
    """
    Class creates a dataloader for training and validation Fre-gan.
    It takes parameters from config.yaml and path to the folder with audios as a parameter.
    Please, look into config.yaml dataset section to find out about parameters in detail.
    """
    _log_file = "bad_samples.txt"

    def __init__(self, config: Dict, mode: str):
        for key, value in config.items():
            setattr(self, key, value)

        tsv_filepath = self.filepath
        self.mode = mode
        self.audio_files = get_file_names(f"{DATA_PATH}/{tsv_filepath}", f"{DATA_PATH}/audio/", self.speaker_id)
        np.random.shuffle(self.audio_files)

        if self.mode == "train":
            self.audio_files = self.audio_files[:int(len(self.audio_files) * self.train_set_size)]
        else:
            self.audio_files = self.audio_files[int(len(self.audio_files) * self.train_set_size):]

        self.wav_files = self._load_audios()
        self._n_samples = len(self.audio_files)

    def _load_audios(self) -> Dict:
        wav_dict = {}
        for i, audio_file_path in enumerate(self.audio_files):
            audio = load_audio(audio_file_path, self.sr, self._log_file)
            audio = librosa.resample(audio, orig_sr=self.sr, target_sr=self.target_sr)
            wav_dict[i] = audio
        return wav_dict

    def __compute_mel_spectrogram(self, audio: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function takes the path to an audio file and returns it's mel spectrogram.
        :param audio: np.ndarray, original audio that will be used for creating mel-spec
        :return: 2 tensors: mel-spectrogram and original audio padded/truncated to the target_audio_length.
        """

        # pad / crop random part from audio signal
        if self.mode != "test":
            audio = pad_crop_audio(audio, self.target_audio_length)

        mel_spectrogram = get_mel_spectrogram(audio, self.hop_size, self.n_mels, self.n_fft, self.power,
                                              self.target_sr, self.f_min, self.f_max, self.normalize_spec)
        return mel_spectrogram, audio

    def __len__(self) -> int:
        if self.dataset_size:
            return self.dataset_size
        return self._n_samples

    def __getitem__(self, idx: int) -> Optional[torch.Tensor, torch.Tensor]:
        mel_spectrogram, audio = self.__compute_mel_spectrogram(self.wav_files[idx])
        return mel_spectrogram, audio


def get_dataloaders(dataset_config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(MelDataset(dataset_config, mode="train"),
                                  batch_size=dataset_config["batch_size"],
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=dataset_config["num_workers"])

    val_dataloader = DataLoader(MelDataset(dataset_config, mode="val"),
                                batch_size=dataset_config["batch_size"],
                                shuffle=False,
                                pin_memory=True,
                                num_workers=dataset_config["num_workers"])

    test_dataloader = DataLoader(MelDataset(dataset_config, mode="test"),
                                 batch_size=1,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=dataset_config["num_workers"])

    return train_dataloader, val_dataloader, test_dataloader
