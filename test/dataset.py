import random

from omegaconf import OmegaConf

from src.dataset import MelDataset


def test_mel_dataset_shapes():
    """
    Test checks mel-spec shape and padded audio shapes of random 100 samples from train dataset matches target.
    """
    config = OmegaConf.load("../src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    dataset_config = config["dataset"]
    dataset = MelDataset(dataset_config, mode="train")

    for _ in range(5):
        assert dataset[random.randint(0, len(dataset))][0].shape == (80, 32), f"Mel-spec shapes mismatch :("
        assert dataset[random.randint(0, len(dataset))][1].shape[0] == dataset_config["target_audio_length"], \
            f"Padded audio shape doesn't match target_audio_length in config :("
