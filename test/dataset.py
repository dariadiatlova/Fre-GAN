import random

from omegaconf import OmegaConf

from src.dataset.dataset import MelDataset


def test_shapes():
    """
    Test checks mel-spec of random 100 audios matches target shape (80, 626).
    """
    config = OmegaConf.load("../src/config.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    dataset_config = config["dataset"]
    dataset = MelDataset(dataset_config, train=True)
    for _ in range(100):
        assert dataset[random.randint(0, len(dataset))].shape == (80, 173), f"Shape mismatch :("
