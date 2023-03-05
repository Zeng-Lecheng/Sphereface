from torch.utils.data import DataLoader
from utils import get_names, LFWDataset
from train import Trainer
from model import Net

if __name__ == '__main__':
    batch_size = 20
    train_name_list = get_names(test=False)
    test_name_list = get_names(test=True)

    train_set = LFWDataset(path='data/lfw', name_list=train_name_list)
    test_set = LFWDataset(path='data/lfw', name_list=test_name_list)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    net = Net(num_class=train_set.num_labels)
    trainer = Trainer(net, train_loader, test_loader)
    trainer.train(epochs=10, lr=.001)
    print()
