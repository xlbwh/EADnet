import torch
from torch import nn


class ResBiTimeNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, seq_len, dropout=0.5, output_size=9):
        super(ResBiTimeNet, self).__init__()
        self.lstm_1 = nn.LSTM(input_size, hidden_sizes[0], num_layers=2, batch_first=True, bidirectional=True,
                              dropout=dropout)
        self.lstm_21 = nn.LSTM(2 * hidden_sizes[0], hidden_sizes[1], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        self.lstm_22 = nn.LSTM(input_size, hidden_sizes[1], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        self.lstm_31 = nn.LSTM(2 * hidden_sizes[1], hidden_sizes[2], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        self.lstm_32 = nn.LSTM(4 * hidden_sizes[1], hidden_sizes[2], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        self.lstm_41 = nn.LSTM(2 * hidden_sizes[2], hidden_sizes[3], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        self.lstm_42 = nn.LSTM(4 * hidden_sizes[2], hidden_sizes[3], num_layers=2, batch_first=True, bidirectional=True,
                               dropout=dropout)
        hidd = 2 * hidden_sizes[0] + 4 * (hidden_sizes[1] + hidden_sizes[2] + hidden_sizes[3])
        self.lstm_5 = nn.LSTM(hidd, hidden_sizes[4], num_layers=2, batch_first=True, bidirectional=True,
                              dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_sizes[4] * seq_len, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x1, _ = self.lstm_1(x)

        x_x1, _ = self.lstm_21(x1)
        x_x2, _ = self.lstm_22(x)
        x2 = torch.cat([x_x1, x_x2], dim=2)

        x_x1, _ = self.lstm_31(x_x1)
        x_x2, _ = self.lstm_32(x2)
        x3 = torch.cat([x_x1, x_x2], dim=2)

        x_x1, _ = self.lstm_41(x_x1)
        x_x2, _ = self.lstm_42(x3)
        x4 = torch.cat([x_x1, x_x2], dim=2)

        x = torch.cat([x1, x2, x3, x4], dim=2)
        x, _ = self.lstm_5(x)

        # fully connected layers:
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
