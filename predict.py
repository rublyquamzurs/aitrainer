
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader

from seek.model import Net
from seek.text_dataset import TextDataset


VECTOR_LEN = 80


class DGAPrediction:
    def __init__(self, data):
        self.device = torch.device("cpu")
        model = Net()
        model.load_state_dict(torch.load("dga_lstm_epoch_50.pt"))
        model.eval()
        self.model = model
        self.data_loader = DataLoader(TextDataset.get_dump_ass(data), batch_size=128)

    def run(self):
        ans = None
        for data, target in self.data_loader:
            data = data.to(self.device)
            output = self.model(data)
            pred: torch.Tensor = output.argmax(dim=1, keepdim=False)
            cur = pred.cpu().detach().numpy()
            if ans is None:
                ans = cur
            else:
                ans = np.concatenate((ans, cur), axis=0)
        return ans


def main():
    data = []
    with open("ai/test/test_domains.csv", "r") as fp:
        reader = csv.reader(fp)
        for record in reader:
            data.append([record[1], 0])
    t_data = np.array(data)

    dp = DGAPrediction(t_data)
    ret = dp.run()
    with open("pred.csv", "w", encoding="utf-8") as fp:
        header = ["id", "domain", "label"]
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for i, r in enumerate(ret):
            one = {
                "id": data[i][0],
                "domain": data[i][1],
                "label": r
            }
            writer.writerow(one)


if __name__ == "__main__":
    main()
