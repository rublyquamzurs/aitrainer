
import torch
import torch.nn.functional as F


class Method:
    def __init__(self):
        pass

    @staticmethod
    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break

    @staticmethod
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        b_is_b = 0
        b_is_w = 0
        w_is_b = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.binary_cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
                actual = target.argmax(dim=1, keepdim=False)
                correct += pred.eq(actual.view_as(pred)).sum().item()

                true_b = torch.eq(actual, 1)
                pred_b = torch.eq(pred, 1)
                true_w = torch.eq(actual, 0)
                pred_w = torch.eq(pred, 0)
                b_is_b += true_b.eq(pred_b).sum().item()
                b_is_w += true_b.eq(pred_w).sum().item()
                w_is_b += true_w.eq(pred_b).sum().item()

        test_loss /= len(test_loader.dataset)

        p1 = b_is_b / (b_is_b + b_is_w)
        p2 = b_is_b / (b_is_b + w_is_b)
        score = p1 * p2 * 2 / (p1 + p2)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Score: {:.2f}%\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), score * 100.))
