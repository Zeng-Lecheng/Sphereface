import os
from warnings import warn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import AverageMeter


class Trainer:

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.trained = False

    def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            lr: float,
    ) -> None:

        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        loss_track = AverageMeter()
        ce = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(epochs):
            loss_track.reset()
            self.model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)

                loss = ce(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=len(data))

            test_acc, test_loss = self.eval(test_loader)
            print(f'Epoch: {epoch}, Test accuracy: {test_acc}, Train loss: {loss_track.avg}, Test loss: {test_loss}')

        self.trained = True
        print("Training completed.")

    def eval(self, test_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        accuracy_counter = AverageMeter()
        loss_tracker = AverageMeter()
        ce = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for x, y_true in test_loader:
                x, y_true = x.to(self.device), y_true.to(self.device)
                y_pred = self.model(x)
                # noinspection PyTypeChecker
                accuracy_counter.update(
                    torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(y_true, dim=1)).item() / len(x),
                    len(x)
                )
                loss_tracker.update(ce(y_pred, y_true).item(), len(x))
        return accuracy_counter.avg, loss_tracker.avg

    def save_model(self, path: str):
        if not self.trained:
            warn(f'Trying to save an untrained model to {path}')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, "mnist.pth"))

    def infer(self, sample: torch.Tensor) -> int:
        if not self.trained:
            warn(f'Trying to infer with an untrained model.')
        sample = sample.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.argmax(self.model(sample)).item()

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(path, 'mnist.pth'), map_location=self.device))
        self.trained = True
