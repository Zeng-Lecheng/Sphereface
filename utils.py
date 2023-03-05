import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def get_names(test: bool = False) -> list[str]:
    res = []
    if not test:
        lines = list(open('data/pairsDevTrain.txt', 'r', encoding='utf-8'))
    else:
        lines = list(open('data/pairsDevTest.txt', 'r', encoding='utf-8'))

    for line in lines[1:]:  # exclude first line
        res.append(line.split()[0]) # names are at the first part of splits

    return res


class LFWDataset(Dataset):
    def __init__(self, path: str, name_list: list[str] , device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.to_tensor = ToTensor()
        self.images: list[Image] = []
        self.labels: list[str] = []
        labels_set: set[str] = set()

        for path_name in os.listdir(path):
            # path_name is literally name of the persons
            if path_name in name_list:
                for single_image in os.listdir(os.path.join(path, path_name)):
                    with open(os.path.join(path, path_name, single_image), 'rb') as f:
                        img = Image.open(f)
                        img.load()  # Address too many open files error
                    img_aug = img.transpose(Image.FLIP_LEFT_RIGHT)
                    self.images.append(img)
                    self.images.append(img_aug)
                    self.labels.append(path_name)
                    self.labels.append(path_name)
                    labels_set.add(path_name)

        self.labels_encoding: dict[str, int] = {}
        for i, name in enumerate(labels_set):
            self.labels_encoding[name] = i
        self.num_labels: int = len(labels_set)

    def __getitem__(self, index):
        image_tensor = self.to_tensor(self.images[index])

        name = self.labels[index]
        label_tensor = torch.tensor([self.labels_encoding[name]], dtype=torch.int64)

        return image_tensor.to(self.device), label_tensor.to(self.device)

    def __len__(self):
        return len(self.images)
