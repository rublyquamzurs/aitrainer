
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from seek.method import Method
from seek.model import Net
from seek.text_dataset import TextDataset


VECTOR_LEN = 80


class Entry:
    def __init__(self, train_data, test_data):
        self.tr = train_data
        self.te = test_data

    def run(self):
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=64, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.97, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--no-mps', action='store_true', default=False,
                            help='disables macOS GPU training')
        parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        args = parser.parse_args()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        use_mps = not args.no_mps and torch.backends.mps.is_available()

        torch.manual_seed(args.seed)

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        train_loader = DataLoader(TextDataset.get_dump_ass(self.tr), **train_kwargs)
        test_loader = DataLoader(TextDataset.get_dump_ass(self.te), **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = ExponentialLR(optimizer, gamma=args.gamma, verbose=True)
        method = Method()
        for epoch in range(1, args.epochs + 1):
            method.train(args, model, device, train_loader, optimizer, epoch)
            method.test(model, device, test_loader)
            scheduler.step()
            if epoch % 10 == 0 and args.save_model:
                torch.save(model.state_dict(), "dga_lstm_epoch_%d.pt" % epoch)

        if args.save_model:
            torch.save(model.state_dict(), "dga_lstm_epoch.pt")
