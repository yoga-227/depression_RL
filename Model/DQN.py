import torch

from torch import nn


class DQN(nn.Module):
    def __init__(self, num_classes):
        super(DQN, self).__init__()
        hidden_size = 88
        rnn_hidden = 128
        num_layers = 2
        dropout = 0.1
        bidirectional = True

        self.lstm = nn.LSTM(hidden_size, rnn_hidden, num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, eGemaps):
        out, _ = self.lstm(eGemaps)
        # out = self.dropout(out)
        last_hs = out[:, -1, :]
        out = self.fc_rnn(last_hs)  # 句子最后时刻的 hidden state
        return out