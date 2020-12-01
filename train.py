
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.bilstm import BiLSTM
from modules.dataset import PPGIMUData
from modules.tblogger import TBLogger


def train(train_input_path, train_label_path, test_input_path, test_label_path, tb_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 3000  # whole signal length (30 sec x 100 Hz)
    batch_size = 50
    lr = 1e-4
    max_epoch = 200

    # Define dataset and data loader
    train_data_loader = DataLoader(PPGIMUData(train_input_path, train_label_path),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    test_data_loader = DataLoader(PPGIMUData(test_input_path, test_label_path),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=True)

    # Define NN
    model = BiLSTM(seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    critetion = nn.CrossEntropyLoss().to(device)

    # Define TensorBoard Logger
    sub_dir = f'{batch_size=}-{lr=}-{max_epoch=}'
    if tb_path:
        tb_path = tb_path / sub_dir
        tensorboard_logger = TBLogger(tb_path)

    # Start epoch
    for epoch in tqdm(range(max_epoch)):
        step_loss_lst = []

        # Start step
        for i, (inputs, targets) in enumerate(train_data_loader):
            model.train()
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            loss = critetion(outputs, targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Record results
            step_loss_lst.append(loss.item())
            if not (i+1)%100 and tb_path:
                tensorboard_logger.scalar_summary(f'epoch_{epoch}/train/loss')



if __name__ == '__main__':
    DATA_PATH = Path('/Volumes/Samsung_SSD/CIME-PPG-dataset-2018')
    train_input_path = DATA_PATH / 'train_input_processed.npy'
    train_label_path = DATA_PATH / 'train_label.npy'
    test_input_path = DATA_PATH / 'test_input_processed.npy'
    test_label_path = DATA_PATH / 'test_label.npy'
    tb_path = Path('./log/')

    train(train_input_path, train_label_path, test_input_path, test_label_path, tb_path)
