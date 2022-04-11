import torch
from omegaconf import OmegaConf

from src.model.modules.generator import RCG
from src.model.modules.period_discriminator import RPD
from src.model.modules.scale_discriminator import RSD


def test_discriminators():
    """
    Test checks that discriminators takes and process input and generated audio without crashing on shapes mismatch.
    """
    config = OmegaConf.load("../src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    target_audio_length = config["dataset"]["target_audio_length"]
    generator = RCG(config["rcg"])
    period_discriminator = RPD(device="cpu", negative_slope=0.1)
    scale_discriminator = RSD(device="cpu", negative_slope=0.1)
    for _ in range(10):
        dummy_audio = torch.rand(1, target_audio_length).unsqueeze(1)
        dummy_spectrogram = torch.rand(1, 80, 32)
        generated_audio = generator(dummy_spectrogram)
        try:
            period_discriminator(dummy_audio, generated_audio)
            scale_discriminator(dummy_audio, generated_audio)
        except Exception as e:
            raise e
