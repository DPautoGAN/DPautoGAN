import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import census_dataset
import credit_dataset
import mimic_dataset

import dp_optimizer
import sampling
import analysis


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

    _, train_dataset, validation_dataset, _ = dataset.get_datasets()
    x_validation = next(iter(DataLoader(validation_dataset, batch_size=len(validation_dataset)))).to(params['device'])

    autoencoder = Autoencoder(
        example_dim=np.prod(train_dataset[0].shape),
        compression_dim=params['compress_dim'],
        binary=params['binary'],
        device=params['device'],
    )

    decoder_optimizer = dp_optimizer.DPAdam(
        l2_norm_clip=params['l2_norm_clip'],
        noise_multiplier=params['noise_multiplier'],
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        params=autoencoder.get_decoder().parameters(),
        lr=params['lr'],
        betas=(params['b1'], params['b2']),
        weight_decay=params['l2_penalty'],
    )

    encoder_optimizer = torch.optim.Adam(
        params=autoencoder.get_encoder().parameters(),
        lr=params['lr'] * params['microbatch_size'] / params['minibatch_size'],
        betas=(params['b1'], params['b2']),
        weight_decay=params['l2_penalty'],
    )

    autoencoder_loss = lambda inp, target: nn.BCELoss(reduction='none')(inp, target).sum(dim=1).mean(dim=0) if params['binary'] else nn.MSELoss()

    print('Achieves ({}, {})-DP'.format(
        analysis.epsilon(
            len(train_dataset),
            params['minibatch_size'],
            params['noise_multiplier'],
            params['iterations'],
            params['delta']
        ),
        params['delta'],
    ))

    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        iterations=params['iterations'],
    )

    iteration = 0
    train_losses, validation_losses = [], []
    for X_minibatch in minibatch_loader(train_dataset):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        for X_microbatch in microbatch_loader(X_minibatch):
            X_microbatch = X_microbatch.to(params['device'])
            decoder_optimizer.zero_microbatch_grad()
            output = autoencoder(X_microbatch)
            loss = autoencoder_loss(output, X_microbatch)
            loss.backward()
            decoder_optimizer.microbatch_step()
        encoder_optimizer.step()
        decoder_optimizer.step()

        validation_loss = autoencoder_loss(autoencoder(x_validation).detach(), x_validation)
        train_losses.append(loss.item())
        validation_losses.append(validation_loss.item())

        if iteration % 100 == 0:
            print ('[Iteration %d/%d] [Loss: %f] [Validation Loss: %f]' % (
                iteration, params['iterations'], loss.item(), validation_loss.item())
            )
        iteration += 1

    return autoencoder, pd.DataFrame(data={'train': train_losses, 'validation': validation_losses})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--b1', type=float, default=0.9, help='decay of first order momentum of gradient for adam (default: 0.5)')
    parser.add_argument('--b2', type=float, default=0.999, help='decay of first order momentum of gradient for adam (default: 0.999)')
    parser.add_argument('--binary', type=bool, default=True, help='whether data type is binary (default: true)')
    parser.add_argument('--compress-dim', type=int, default=128, help='compression dimension of the autoencoder (default: 128)')
    parser.add_argument('--dataset', type=str, default='mimic', help='the dataset to be used for training (default: mimic)')
    parser.add_argument('--delta', type=float, default=1.2871523321606923e-5, help='delta for epsilon calculation (default: ~1e-5)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--iterations', type=int, default=30000, help='number of iterations to train (default: 30000)')
    parser.add_argument('--l2-norm-clip', type=float, default=0.8157, help='upper bound on the l2 norm of gradient updates (default: 0.8157)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--microbatch-size', type=int, default=1, help='input microbatch size for training (default: 1)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='input minibatch size for training (default: 100)')
    parser.add_argument('--noise-multiplier', type=float, default=1.1, help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    params = vars(parser.parse_args())

    model, losses = train(params)

    with open('dp_autoencoder.dat', 'wb') as f:
        torch.save(model, f)

    losses.plot()
    plt.savefig('dp_autoencoder.png')
    losses.to_csv('dp_autoencoder.csv')

