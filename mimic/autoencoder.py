import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import census_dataset
import credit_dataset
import mimic_dataset


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)


class Autoencoder(nn.Module):
    def __init__(self, example_dim, compression_dim, binary, device='cpu'):
        super(Autoencoder, self).__init__()
        self.compression_dim = compression_dim

        self.encoder = nn.Sequential(
            nn.Linear(example_dim, compression_dim),
            nn.Tanh() if binary else nn.ReLU()
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, example_dim),
            nn.Sigmoid() if binary else nn.ReLU()
        ).to(device)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_compression_dim(self):
        return self.compression_dim


def train(params):
    dataset = {
        'mimic': mimic_dataset,
        'credit': credit_dataset,
        'census': census_dataset,
    }[params['dataset']]

    train_dataset, _, validation_dataset, _ = dataset.get_datasets()
    x_validation = next(iter(DataLoader(validation_dataset, batch_size=len(validation_dataset)))).to(params['device'])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
    )

    autoencoder = Autoencoder(
        example_dim=np.prod(next(iter(train_dataloader))[0].shape),
        compression_dim=params['compress_dim'],
        binary=params['binary'],
        device=params['device']
    )

    optimizer = torch.optim.Adam(
        params=autoencoder.parameters(),
        lr=params['lr'],
        betas=(params['b1'], params['b2']),
        weight_decay=params['l2_penalty'],
    )

    autoencoder_loss = lambda inp, target: nn.BCELoss(reduction='none')(inp, target).sum(dim=1).mean(dim=0) if params['binary'] else nn.MSELoss()

    train_losses, validation_losses = [], []
    for epoch in range(params['epochs']):
        for i, x in enumerate(train_dataloader):
            x = x.to(params['device'])

            autoencoder.zero_grad()
            output = autoencoder(x)
            loss = autoencoder_loss(output, x)
            loss.backward()
            optimizer.step()

            validation_loss = autoencoder_loss(autoencoder(x_validation).detach(), x_validation)

            train_losses.append(loss.item())
            validation_losses.append(validation_loss.item())

            batches_done = epoch * len(train_dataloader) + i
            if batches_done % 10 == 0:
                print ('[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Validation Loss: %f]' % (
                    epoch, params['epochs'], i, len(train_dataloader), loss.item(), validation_loss.item())
                )

    return autoencoder, pd.DataFrame(data={'train': train_losses, 'validation': validation_losses})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--b1', type=float, default=0.9, help='decay of first order momentum of gradient for adam (default: 0.5)')
    parser.add_argument('--b2', type=float, default=0.999, help='decay of first order momentum of gradient for adam (default: 0.999)')
    parser.add_argument('--compress-dim', type=int, default=128, help='compression dimension of the autoencoder (default: 128)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--binary', type=bool, default=True, help='whether data type is binary (default: true)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    parser.add_argument('--dataset', type=str, default='mimic', help='the dataset to be used for training (default: mimic)')
    params = vars(parser.parse_args())

    model, losses = train(params)

    with open('autoencoder.dat', 'wb') as f:
        torch.save(model, f)

    losses.plot()
    plt.savefig('autoencoder.png')

