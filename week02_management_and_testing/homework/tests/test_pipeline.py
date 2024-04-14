import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step
from modeling.unet import UnetModel
from modeling.training import generate_samples, train_epoch


@pytest.fixture
def train_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    torch.random.manual_seed(42)

    unet_model = UnetModel(3, 3, hidden_size=32)
    unet_model.to(device)
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=unet_model,
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))

    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, tmp_path, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    torch.random.manual_seed(42)

    if device == "cpu":
        # take a small subset of the dataset to speed up the test
        train_dataset = torch.utils.data.Subset(train_dataset, range(1000))

    unet_model = UnetModel(3, 3, hidden_size=32)
    unet_model.to(device)
    ddpm = DiffusionModel(
        eps_model=unet_model,
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-3)
    dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    train_epoch(ddpm, dataloader, optim, device)
    generate_samples(ddpm, device, tmp_path / "samples.png")
