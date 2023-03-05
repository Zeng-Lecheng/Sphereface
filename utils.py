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


class LFWDataset(Dataset):
    def __init__(self, img_path: str, name_path: str , device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.to_tensor = ToTensor()
        self.images: list[Image] = []
        self.labels: list[str] = []

        name_set: set[str] = self.get_names(name_path)
        for path_name in name_set:
            # path_name is literally name of the persons
            for single_image in os.listdir(os.path.join(img_path, path_name)):
                with open(os.path.join(img_path, path_name, single_image), 'rb') as f:
                    img = Image.open(f)
                    img.load()  # Address too many open files error
                img_aug = img.transpose(Image.FLIP_LEFT_RIGHT)
                self.images.append(img)
                self.images.append(img_aug)
                self.labels.append(path_name)
                self.labels.append(path_name)

        self.labels_encoding: dict[str, int] = {}
        for i, name in enumerate(name_set):
            self.labels_encoding[name] = i
        self.num_labels: int = len(name_set)

    def __getitem__(self, index):
        image_tensor = self.to_tensor(self.images[index])

        name = self.labels[index]
        label_tensor = torch.tensor([self.labels_encoding[name]], dtype=torch.int64)

        return image_tensor.to(self.device), label_tensor.to(self.device)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_names(path: str) -> set[str]:
        res = []
        lines = list(open(path, 'r', encoding='utf-8'))

        for line in lines[1:]:  # exclude first line
            res.append(line.split()[0])
            if line.split()[2].isalpha():
                res.append(line.split()[2])     # the mismatched samples

        return set(res)


class LFWPairDataset(Dataset):
    def __init__(self, img_path: str, name_path: str , device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.to_tensor = ToTensor()
        self.images: list[tuple[Image, Image, bool]] = []

        name_list = self.get_names(name_path)

        for item in name_list:
            if len(item) == 3:  # matched pair
                id1 = '0' * (4 - len(item[1])) + item[1]    # filling leading zeros in file name
                id2 = '0' * (4 - len(item[2])) + item[2]
                name = item[0]
                with open(os.path.join(img_path, name, f'{name}_{id1}.jpg'), 'rb') as f:
                    img1 = Image.open(f)
                    img1.load()
                with open(os.path.join(img_path, name, f'{name}_{id2}.jpg'), 'rb') as f:
                    img2 = Image.open(f)
                    img2.load()
                self.images.append((img1, img2, True))
            else:
                name1 = item[0]
                id1 = '0' * (4 - len(item[1])) + item[1]
                name2 = item[2]
                id2 = '0' * (4 - len(item[3])) + item[3]
                with open(os.path.join(img_path, name1, f'{name1}_{id1}.jpg'), 'rb') as f:
                    img1 = Image.open(f)
                    img1.load()
                with open(os.path.join(img_path, name2, f'{name2}_{id2}.jpg'), 'rb') as f:
                    img2 = Image.open(f)
                    img2.load()
                self.images.append((img1, img2, False))


    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image1_tensor = self.to_tensor(self.images[index][0])
        image2_tensor = self.to_tensor(self.images[index][1])
        label_tensor = torch.tensor(self.images[index][2])
        return image1_tensor.to(self.device), image2_tensor.to(self.device), label_tensor.to(self.device)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_names(path: str) -> list[tuple]:
        res = []
        lines = list(open(path, 'r', encoding='utf-8'))

        for line in lines[1:]:  # exclude first line
            res.append(tuple(line.split()))  # names are at the first part of splits

        return res
