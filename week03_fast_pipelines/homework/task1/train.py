import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dynamic_scaling: bool = False
) -> None:
    model.train()

    S = 2**16 # init value was taken from official torch.amp.GradScaler class

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            outputs = model(images)
            assert outputs.dtype == torch.float16
            loss = criterion(outputs, labels)
        
        loss *= S
        loss.backward()


        optimizer.step()        
    
        if dynamic_scaling:
            S /= 2

        optimizer.zero_grad()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train():
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device)


if __name__ == '__main__':
    train()
