import torch
from omegaconf import OmegaConf

from src.model.generator import RCG
from src.model.period_discriminator import RPD
from src.model.scale_discriminator import RSD


def test_discriminators():
    """
    Test checks that discriminators takes and process input and generated audio without crashing on shapes mismatch.
    """
    config = OmegaConf.load("../src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    target_audio_length = config["dataset"]["target_audio_length"]
    generator = RCG(config["rcg"])
    period_discriminator = RPD()
    scale_discriminator = RSD()
    for _ in range(10):
        dummy_audio = torch.rand(1, target_audio_length)
        dummy_spectrogram = torch.rand(1, 80, 173)
        generated_audio = generator(dummy_spectrogram).squeeze(1)
        try:
            period_discriminator(dummy_audio, generated_audio)
            scale_discriminator(dummy_audio, generated_audio)
        except Exception as e:
            raise e
