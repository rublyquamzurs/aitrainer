
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(input_size=8,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(80 * 64, 2)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x
