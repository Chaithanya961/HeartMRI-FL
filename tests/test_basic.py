import torch

from heartmri_fl import __version__
from heartmri_fl.data import build_synthetic_client_data
from heartmri_fl.model import SimpleHeartMRIAutoencoder


def test_version():
    assert __version__ == "0.1.0"


def test_synthetic_client_data_builds():
    clients = build_synthetic_client_data(num_hospitals=2, subclients_per_hospital=2, samples_per_client=8)
    assert len(clients) == 4
    for dataset in clients.values():
        assert len(dataset.train_dataset) > 0
        assert len(dataset.val_dataset) > 0


def test_autoencoder_forward():
    model = SimpleHeartMRIAutoencoder()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    assert y.shape == x.shape
