import argparse
import soundfile as sf

from typing import Dict
from omegaconf import OmegaConf
from pathlib import Path

from data import DATA_PATH
from src.model.lightning_model import FreGan
from src.utils import load_audio, pad_input_audio_signal, get_mel_spectrogram


def feature_extraction(params: Dict):
    audio_data = load_audio(params.audio_file_path, params.sample_rate)
    padded_signal = pad_input_audio_signal(audio_data, params.segment_size)
    mel_spectrogram = get_mel_spectrogram(padded_signal, hop_length=256, n_mels=80, n_fft=1024,
                                          sample_rate=params.sample_rate)
    return mel_spectrogram


def generate_audio(params: Dict, train_config: Dict) -> None:
    config = OmegaConf.load(train_config)
    config = OmegaConf.to_container(config, resolve=True)
    model_weights = params.model_weights_path
    model = FreGan.load_from_checkpoint(checkpoint_path=model_weights, config=config)
    model.eval()
    mel_spectrogram = feature_extraction(params)
    generated_audio = model(mel_spectrogram.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
    if max(abs(generated_audio)) > 1:
        generated_audio /= max(abs(generated_audio))
    sf.write(params.output_wav_path, generated_audio, params.sample_rate)
    return


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', '--audio_file_path',
                        help='Path to the wav file to evaluate.',
                        type=str)
    parser.add_argument('-w', '--model_weights_path',
                        help='Path to the .ckpt file to initialize model weights.',
                        type=str)
    parser.add_argument('-o', '--output_wav_path',
                        help='Path of the generated audio, default is data/generated_samples/generated.wav',
                        type=str,
                        default=Path(f"{DATA_PATH}/generated_samples/generated.wav"))
    parser.add_argument('-sr', '--sample_rate',
                        help='Sample rate of the input audio file. Default is 4100.',
                        type=str,
                        default=44100)
    parser.add_argument('-s', '--segment_size',
                        help='Size of the output audio file (sr * seconds). Default is 44100 * 3 = 132300',
                        type=int,
                        default=132300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    params = parser.parse_args()
    generate_audio(params, "src/config.yaml")
    print(f"Audio file was successfully generated and saved into: {params.output_wav_path}")
