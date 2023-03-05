import os
import time
from warnings import warn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model import AngleLoss
from utils import AverageMeter


class Trainer:

    def __init__(
            self, model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.trained = False

    def train(
            self,
            epochs: int,
            lr: float,
    ) -> None:

        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        loss_track = AverageMeter()
        cri = AngleLoss().to(self.device)

        for epoch in range(epochs):
            loss_track.reset()
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)

                loss = cri(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=len(data))

            print(f'Epoch: {epoch}, Train loss: {loss_track.avg}')

        self.trained = True
        print("Training completed.")

    def eval(self, test_loader: DataLoader) -> list[tuple]:
        print('Evaluating...')
        self.model.eval()
        accuracy_counter = AverageMeter()
        accuracies: list[tuple[float, float]] = []  # list of (thres, accuracy)
        with torch.no_grad():
            for threshold in torch.linspace(-1., 1., 20):
                print(f'Evaluating threshold {threshold}')
                for x1, x2, y in test_loader:
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    x1_feat = self.model(x1, get_feature=True)
                    x2_feat = self.model(x2, get_feature=True)
                    # noinspection PyTypeChecker
                    cos_sim = torch.cosine_similarity(x1_feat, x2_feat)
                    accuracy_counter.update(
                        torch.sum(torch.logical_not(torch.logical_xor(cos_sim > threshold, y))).item() / len(x1),
                        len(x1)
                    )
                accuracies.append((threshold.item(), accuracy_counter.avg))
        return accuracies

    def save_model(self, path: str):
        if not self.trained:
            warn(f'Trying to save an untrained model to {path}')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, time.strftime('%Y%m%d_%H%M%S.pt')))

    def infer(self, sample: torch.Tensor) -> int:
        if not self.trained:
            warn(f'Trying to infer with an untrained model.')
        sample = sample.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.argmax(self.model(sample)).item()

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(path), map_location=self.device))
        self.trained = True
