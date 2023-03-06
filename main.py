from torch.utils.data import DataLoader
from utils import LFWDataset, LFWPairDataset
from train import Trainer
from model import Net

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 20

    train_set = LFWDataset(img_path='data/lfw', name_path='data/pairsDevTrain.txt', device=device)
    test_set = LFWPairDataset(img_path='data/lfw', name_path='data/pairsDevTest.txt', device=device)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    net = Net(num_class=train_set.num_labels)
    trainer = Trainer(net, train_loader, test_loader, device=device)
    trainer.train(epochs=20, lr=.00005)
    trainer.save_model('saved_models/')
    threshold, accuracy = trainer.eval(test_loader)
    print(f'Accuracy: {accuracy} at threshold {threshold}')
