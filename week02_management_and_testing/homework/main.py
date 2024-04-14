import torch
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main(device: str):
    initialize(config_path='./configs')
    configs = compose(config_name="start_config")

    print(configs.model)
    wandb.init(
        project="my-awesome-project",

        config={
        "betas": (configs.model.beta1, configs.model.beta2),
        "timesteps_num": configs.model.timesteps,
        "learning_rate": configs.model.learning_rate,
        "architecture": configs.model.name,
        "dataset": configs.dataset.name,
        "epochs": configs.model.epochs,
        "batch_size": configs.model.batch_size
        }
    )

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(configs.model.beta1, configs.model.beta2),
        num_timesteps=configs.model.timesteps,
    )
    ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=configs.model.batch_size, \
                            num_workers=configs.model.num_workers, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=configs.model.learning_rate)

    for i in range(configs.model.epochs):
        train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
