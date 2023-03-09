from torch.utils.data import DataLoader
from utils import LFWDataset, LFWPairDataset
from train import Trainer
from model import Net
import argparse


def parse_args():
    parser = argparse.ArgumentParser('mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=2e-4, help="learning rate")
    parser.add_argument('--bs', type=int, default=20, help="batch size")
    parser.add_argument('--d', type=str, default='cuda', help='running device, cpu or cuda')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.bs
    device = args.d
    learning_rate = args.lr
    epochs = args.epochs

    train_set = LFWDataset(img_path='data/lfw', name_path='data/pairsDevTrain.txt', device=device)
    test_set = LFWPairDataset(img_path='data/lfw', name_path='data/pairsDevTest.txt', device=device)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    net = Net(num_class=train_set.num_labels)
    trainer = Trainer(net, train_loader, test_loader, device=device)
    trainer.train(epochs=epochs, lr=learning_rate)
    trainer.save_model('saved_models/')
    threshold, accuracy = trainer.eval(test_loader)
    print(f'Accuracy: {accuracy}')
