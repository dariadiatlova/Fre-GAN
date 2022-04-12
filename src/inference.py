import argparse
import re

import librosa
import torch

from omegaconf import OmegaConf
from typing import Dict, Optional

from data.generated_samples import GENERATED_DIR_PATH
from src.model.lightning_model import FreGan
from src.model.melspectrogram import mel_spectrogram
from src.utils import load_audio, pad_input_audio_signal, write_wav_file


def feature_extraction(params: Dict, target_sr: int, config: Dict):
    audio = load_audio(params.audio_file_path, params.sample_rate)
    audio = librosa.resample(audio, orig_sr=params.sample_rate, target_sr=target_sr)
    norm_audio = librosa.util.normalize(audio)
    audio = torch.FloatTensor(norm_audio).type(torch.FloatTensor).unsqueeze(0)
    if params.segment_size:
        audio = pad_input_audio_signal(audio, params.target_audio_length)
    else:
        audio = torch.FloatTensor(audio)
    mel = mel_spectrogram(audio, config["n_fft"], config["n_mels"], config["target_sr"], config["hop_size"],
                          config["win_si1ze"], config["f_min"], config["f_max"])
    return mel.squeeze()


def generate_audio(params: Dict, train_config: Dict) -> str:
    config = OmegaConf.load(train_config)
    config = OmegaConf.to_container(config, resolve=True)
    model_weights = params.model_weights_path
    model = FreGan.load_from_checkpoint(checkpoint_path=model_weights, config=config, inference=True, val_loader=None)
    model.eval()
    model.generator.remove_weight_norm()
    mel_spectrogram = feature_extraction(params, config["dataset"]["target_sr"], config["dataset"])
    generated_audio = model(mel_spectrogram.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
    if params.output_wav_path:
        output_wav_path = params.output_wav_path
    else:
        pattern = re.compile("[^\/]+$")
        filename = pattern.search(params.audio_file_path).group()
        output_wav_path = GENERATED_DIR_PATH / f"generated_{filename}"
    write_wav_file(generated_audio, output_wav_path, config["dataset"]["target_sr"])
    return output_wav_path


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', '--audio_file_path',
                        help='Path to the wav file to evaluate.',
                        type=str)
    parser.add_argument('-w', '--model_weights_path',
                        help='Path to the .ckpt file to initialize model weights.',
                        type=str)
    parser.add_argument('-o', '--output_wav_path',
                        help='Path of the generated audio, default is data/generated_samples/'
                             '<initial_audio_name>_generated.wav',
                        type=Optional[str],
                        default=None)
    parser.add_argument('-sr', '--sample_rate',
                        help='Sample rate of the input audio file. Default is 48000.',
                        type=str,
                        default=48000)
    parser.add_argument('-s', '--segment_size',
                        help='Size of the output audio file (sr * seconds). '
                             'Default is the original audio length.',
                        type=Optional[int],
                        default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    params = parser.parse_args()
    output_wav_path = generate_audio(params, "src/config.yaml")
    print(f"Audio file was successfully generated and saved into: {output_wav_path}")
