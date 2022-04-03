import torch
from omegaconf import OmegaConf

from src.model.modules.generator import RCG


def test_generator_shapes():
    """
    Test checks that generator reproduces from mel-spectrogram audio of the target_audio_length shape.
    """
    config = OmegaConf.load("../src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    target_audio_length = config["dataset"]["target_audio_length"]
    generator = RCG(config["rcg"])
    for _ in range(10):
        dummy_spectrogram = torch.rand(1, 80, 29)
        generated_audio = generator(dummy_spectrogram)
        assert generated_audio.shape[-1] == target_audio_length, f"Expected generator to generate audio of length:" \
                                                                 f"{target_audio_length}, got an audio of length:" \
                                                                 f"{generated_audio.shape[-1]} instead."
