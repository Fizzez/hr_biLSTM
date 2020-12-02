
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
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
    phases = ['train', 'val']

    # Define dataset and data loader
    train_data = PPGIMUData(train_input_path, train_label_path)
    test_data = PPGIMUData(test_input_path, test_label_path)
    train_indices, val_indices = train_test_split(list(range(len(train_data))), test_size=0.2, shuffle=True)

    train_set_loader = DataLoader(Subset(train_data, train_indices),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)
    val_set_loader = DataLoader(Subset(train_data, val_indices),
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)
    test_set_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

    train_val_set_loaders = dict(train=train_set_loader, val=val_set_loader)

    # Define NN
    model = BiLSTM(seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    critetion = nn.CrossEntropyLoss().to(device)

    # Define TensorBoard Logger
    sub_dir = f'{batch_size=} {lr=} {max_epoch=} {datetime.now()}'
    if tb_path:
        tb_path = tb_path / sub_dir
        tensorboard_logger = TBLogger(tb_path)

    # Start epoch
    step_counter_train = 0
    time_start = datetime.now()
    for epoch in range(max_epoch):

        # train, val phase in an epoch
        for phase in phases:
            step_loss_lst = []

            # Start step
            for i, (inputs, targets) in tqdm(enumerate(train_val_set_loaders[phase])):
                with torch.set_grad_enabled(phase == 'train'):
                    model.train() if phase == 'train' else model.eval()
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    targets = torch.flatten(targets).to(device).long()
                    loss = critetion(outputs, targets)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                # Record results
                step_loss_lst.append(loss.item())
                if not (i+1) % 10 and tb_path:
                    tensorboard_logger.scalar_summary(f'epoch_{epoch+1}/{phase}/loss', loss.item(), i+1)
                    if phase == 'train':
                        tensorboard_logger.scalar_summary(
                            f'overall-step/train/loss', loss.item(), step_counter_train+i+1)
                        print(f'Epoch {epoch}/{max_epoch}, Step {i+1}/{len(train_set_loader)}, Loss {loss.item()}, '
                              f'Time {datetime.now()-time_start}')

            step_counter_train += len(train_set_loader)

            if tb_path:
                tensorboard_logger.scalar_summary(f'overall-epoch/{phase}/loss/mean', np.mean(step_loss_lst), epoch+1)
                tensorboard_logger.scalar_summary(f'overall-epoch/{phase}/loss/std', np.std(step_loss_lst), epoch+1)


if __name__ == '__main__':
    DATA_PATH = Path('/Volumes/Samsung_SSD/CIME-PPG-dataset-2018')
    train_input_path = DATA_PATH / 'train_input_processed.npy'
    train_label_path = DATA_PATH / 'train_label.npy'
    test_input_path = DATA_PATH / 'test_input_processed.npy'
    test_label_path = DATA_PATH / 'test_label.npy'
    tb_path = Path('./log/')

    train(train_input_path, train_label_path, test_input_path, test_label_path, tb_path)
