import math

import torch

from torch import nn
import torch.nn.functional as F


class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.1, bias=True):  # sigma_zero表示噪声的标准差的初始值
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)  # 对噪声标准差进行初始化，使其与输入特征数量相关联，能更好地适应不同规模的输入
        self.sigma_weigt = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        # 权重的噪声标准差，维度为2*256，使用sigma_init填充
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        # 存储从标准正态分布中抽取的随机数，用来计算噪声，不会通过反向传播和优化器进行更新，而是通过前向传播重新生成随机数
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))  # 与偏置相关的噪声标准差，通过反向传播更新

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))  # 保留x中元素的符号，将其缩放为绝对值的平方根

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()

        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weigt * noise_v, bias)


class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(LSTMModel, self).__init__()
        hidden_size = 88
        rnn_hidden = 128
        num_layers = 2
        dropout = 0.1
        bidirectional = True

        self.lstm = nn.LSTM(hidden_size, rnn_hidden, num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(2 * rnn_hidden, num_classes)
        # self.advantage = nn.Linear(256, num_classes)
        # self.value = nn.Linear(256, 1)

    def forward(self, eGemaps):
        out, _ = self.lstm(eGemaps)
        # out = self.dropout(out)
        last_hs = out[:, -1, :]
        out = self.fc_rnn(last_hs)  # 句子最后时刻的 hidden state
        return out
        # value = self.value(last_hs)
        # advantage = self.advantage(last_hs)
        # return value + advantage - advantage.mean()


class Noisy_LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(Noisy_LSTMModel, self).__init__()
        hidden_size = 88
        rnn_hidden = 128
        num_layers = 2
        dropout = 0.1
        bidirectional = True

        self.lstm = nn.LSTM(hidden_size, rnn_hidden, num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        # self.fc_rnn = nn.Linear(2 * rnn_hidden, num_classes)
        self.advantage = NoisyFactorizedLinear(256, num_classes)
        self.value = NoisyFactorizedLinear(256, 1)

    def forward(self, eGemaps):
        out, _ = self.lstm(eGemaps)
        # out = self.dropout(out)
        last_hs = out[:, -1, :]
        out = self.fc_rnn(last_hs)  # 句子最后时刻的 hidden state
        return out
        # value = self.value(last_hs)
        # advantage = self.advantage(last_hs)
        # return value + advantage - advantage.mean()


class depression_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(depression_LSTM, self).__init__()
        input_dim = 88
        hidden_dim = 128
        num_layers = 2
        dropout = 0.1
        bidirectional = True

        # 1维卷积层
        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        # self.batch_norm = nn.BatchNorm1d(64)
        self.maxPooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=input_dim, kernel_size=3, padding=1)

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)

        # self.dropout = nn.Dropout(dropout)

        # 全连接层
        # self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向LSTM输出会concatenate，所以乘以2
        # Dueling DQN
        self.advantage = nn.Linear(hidden_dim * 2, num_classes)
        self.value = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # 输入维度：(batch_size, seq_len, input_dim)

        # 1维卷积层
        x_conv = self.conv1d_1(x.permute(0, 2, 1))  # 维度变换为(batch_size, input_dim, seq_len)
        # x_conv = self.batch_norm(x_conv)
        x_conv = nn.ReLU()(x_conv)
        x_conv = self.maxPooling(x_conv)
        x_conv = self.conv1d_2(x_conv)
        # 将卷积结果与原始特征级联
        x_concat = torch.cat((x, x_conv.permute(0, 2, 1)), dim=2)  # 维度变换为(batch_size, seq_len, input_dim + hidden_dim)
        # x_concat = torch.add(x, x_conv.permute(0, 2, 1))

        # 双向LSTM层
        lstm_out, _ = self.lstm(x_concat)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        # out = self.fc(lstm_out)
        #
        # return out

        value = self.value(lstm_out)
        advantage = self.advantage(lstm_out)
        return value + advantage - advantage.mean()


class depression_CNN(nn.Module):
    def __init__(self, num_classes):
        super(depression_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(2 * 40 * 64, 512)
        # self.fc5 = nn.Linear(512, num_classes)
        self.advantage = nn.Linear(512, num_classes)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc4(x))
        # return self.fc5(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()

