
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from seek.method import Method
from seek.model import Net
from seek.text_dataset import TextDataset


VECTOR_LEN = 128


class Entry:
    def __init__(self, train_data, test_data):
        self.trd, self.trl = self._transform(train_data)
        self.ted, self.tel = self._transform(test_data)

    def _transform(self, data):
        count = data.shape[0]
        t = np.zeros(shape=[count, 1, VECTOR_LEN], dtype=np.float32)
        lb = np.zeros(shape=count, dtype=np.int32)
        for i in range(count):
            t[i] = self.__get_flat_vector(data[i][0])
            lb[i] = int(data[i][1])
        return t, lb

    def __get_flat_vector(self, domain: str):
        out = np.zeros(shape=[1, VECTOR_LEN], dtype=np.float32)
        if len(domain) > VECTOR_LEN:
            raise ValueError("domain %s is beyond %d" % (domain, VECTOR_LEN))
        for i, c in enumerate(domain):
            out[0][i] = ord(c)
        return out / np.sqrt(np.sum(np.power(out, 2)))

    def run(self):
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=64, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=5.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
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

        dataset3 = TextDataset(self.trd, self.trl)
        dataset4 = TextDataset(self.ted, self.tel)
        train_loader = DataLoader(dataset3, **train_kwargs)
        test_loader = DataLoader(dataset4, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        method = Method()
        for epoch in range(1, args.epochs + 1):
            method.train(args, model, device, train_loader, optimizer, epoch)
            method.test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
