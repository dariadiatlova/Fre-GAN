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
    dataset = MelDataset(dataset_config, train=True)
    print(dataset[0][0].shape, dataset[0][1].shape)
    for _ in range(100):
        assert dataset[random.randint(0, len(dataset))][0].shape == (80, 29), f"Mel-spec shapes mismatch :("
        assert dataset[random.randint(0, len(dataset))][1].shape == dataset_config["target_audio_length"], \
            f"Padded audio shape doesn't match target_audio_length in config :("
