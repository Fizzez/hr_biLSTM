"""
Definition of bi-directional LSTM model
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, seq_len):
        super(BiLSTM, self).__init__()
        self.input_size = 3     # feature arr dimension is 3 (ppg, x_acce, y_gyro)
        self.seq_len = seq_len
        self.lstm_hidden_size_lst = [100, 50, 50, 20]

        self.lstm_1 = nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[0]//2,
                              num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(self.lstm_hidden_size_lst[0], hidden_size=self.lstm_hidden_size_lst[1]//2,
                              num_layers=1, bidirectional=True)
        self.lstm_3 = nn.LSTM(self.lstm_hidden_size_lst[1], hidden_size=self.lstm_hidden_size_lst[2]//2,
                              num_layers=1, bidirectional=True)
        self.lstm_4 = nn.LSTM(self.lstm_hidden_size_lst[2], hidden_size=self.lstm_hidden_size_lst[3]//2,
                              num_layers=1, bidirectional=True)
        self.fc = nn.Linear(self.lstm_hidden_size_lst[3], 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out_lstm_1, _ = self.lstm_1(x)
        out_lstm_2, _ = self.lstm_2(out_lstm_1)
        out_lstm_3, _ = self.lstm_3(out_lstm_2)
        out_lstm_4, _ = self.lstm_4(out_lstm_3)
        out_fc = self.fc(out_lstm_4)
        out_softmax = self.softmax(out_fc)
        out = torch.flatten(out_softmax, start_dim=0, end_dim=1)
        return out
