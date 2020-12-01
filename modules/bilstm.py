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

        # self.lstm_1 = nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[0],
        #                       num_layers=1, bidirectional=True)
        # self.lstm_2 = nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[1],
        #                       num_layers=1, bidirectional=True)
        # self.lstm_3 = nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[2],
        #                       num_layers=1, bidirectional=True)
        # self.lstm_4 = nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[3],
        #                       num_layers=1, bidirectional=True)

        self.multi_layer_bilstm = nn.Sequential(
            nn.LSTM(self.input_size, hidden_size=self.lstm_hidden_size_lst[0],
                    num_layers=1, bidirectional=True),
            nn.LSTM(self.lstm_hidden_size_lst[0], hidden_size=self.lstm_hidden_size_lst[1],
                    num_layers=1, bidirectional=True),
            nn.LSTM(self.lstm_hidden_size_lst[1], hidden_size=self.lstm_hidden_size_lst[2],
                    num_layers=1, bidirectional=True),
            nn.LSTM(self.lstm_hidden_size_lst[2], hidden_size=self.lstm_hidden_size_lst[3],
                    num_layers=1, bidirectional=True)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        out_lstm, _ = self.multi_layer_bilstm(x)
        out_softmax = self.softmax(out_lstm)
        return torch.where(out_softmax >= 0.5, 1, 0)
