import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dp_autoencoder import Autoencoder
import census_dataset
import credit_dataset
import mimic_dataset

import dp_optimizer
import sampling
import analysis


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, binary, device='cpu'):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, Activation, device):
            return nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=False),
                Activation(),
            ).to(device)

        self.block_0 = block(input_dim, 128, nn.ReLU, device)
        self.block_1 = block(128, output_dim, (nn.Tanh if binary else nn.ReLU), device)

    def forward(self, x):
        x = self.block_0(x) + x
        x = self.block_1(x) + x
        return x



class Discriminator(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

    def forward(self, x):
        return self.model(x)


def train(params):
    dataset = {
        'mimic': mimic_dataset,
        'credit': credit_dataset,
        'census': census_dataset,
    }[params['dataset']]

    _, train_dataset, _, _ = dataset.get_datasets()

    with open('dp_autoencoder.dat', 'rb') as f:
        autoencoder = torch.load(f)

    decoder = autoencoder.get_decoder()

    generator = Generator(
        input_dim=params['latent_dim'],
        output_dim=autoencoder.get_compression_dim(),
        binary=params['binary'],
        device=params['device'],
    )

    g_optimizer = torch.optim.RMSprop(
        params=generator.parameters(),
        lr=params['lr'],
        alpha=params['alpha'],
        weight_decay=params['l2_penalty'],
    )

    discriminator = Discriminator(
        input_dim=np.prod(train_dataset[0].shape),
        device=params['device'],
    )

    d_optimizer = dp_optimizer.DPRMSprop(
        l2_norm_clip=params['l2_norm_clip'],
        noise_multiplier=params['noise_multiplier'],
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        params=discriminator.parameters(),
        lr=params['lr'],
        alpha=params['alpha'],
        weight_decay=params['l2_penalty'],
    )

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
        params['minibatch_size'],
        params['microbatch_size'],
        params['iterations'],
    )

    iteration = 0
    for X_minibatch in minibatch_loader(train_dataset):
        d_optimizer.zero_grad()
        for real in microbatch_loader(X_minibatch):
            real = real.to(params['device'])
            z = torch.randn(real.size(0), params['latent_dim'], device=params['device'], requires_grad=False)
            fake = decoder(generator(z)).detach()

            d_optimizer.zero_microbatch_grad()
            d_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
            d_loss.backward()
            d_optimizer.microbatch_step()
        d_optimizer.step()

        for parameter in discriminator.parameters():
            parameter.data.clamp_(-params['clip_value'], params['clip_value'])

        if iteration % params['d_updates'] == 0:
            z = torch.randn(X_minibatch.size(0), params['latent_dim'], device=params['device'], requires_grad=False)
            fake = decoder(generator(z))

            g_optimizer.zero_grad()
            g_loss = -torch.mean(discriminator(fake))
            g_loss.backward()
            g_optimizer.step()

        if iteration % 100 == 0:
            print('[Iteration %d/%d] [D loss: %f] [G loss: %f]' % (iteration, params['iterations'], d_loss.item(), g_loss.item()))
        iteration += 1

    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.99, help='smoothing parameter for RMS prop (default: 0.99)')
    parser.add_argument('--binary', type=bool, default=True, help='whether data type is binary (default: true)')
    parser.add_argument('--clip-value', type=float, default=0.01, help='upper bound on weights of the discriminator (default: 0.01)')
    parser.add_argument('--d-updates', type=int, default=2, help='number of iterations to update discriminator per generator update (default: 2)')
    parser.add_argument('--dataset', type=str, default='mimic', help='the dataset to be used for training (default: mimic)')
    parser.add_argument('--delta', type=float, default=1.2871523321606923e-5, help='delta for epsilon calculation (default: ~1e-5)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--iterations', type=int, default=30000, help='number of iterations to train (default: 30000)')
    parser.add_argument('--l2-norm-clip', type=float, default=0.35, help='upper bound on the l2 norm of gradient updates (default: 0.35)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    parser.add_argument('--latent-dim', type=int, default=128, help='dimensionality of the latent space (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--microbatch-size', type=int, default=1, help='input microbatch size for training (default: 10)')
    parser.add_argument('--minibatch-size', type=int, default=1000, help='input minibatch size for training (default: 1000)')
    parser.add_argument('--noise-multiplier', type=float, default=1.1, help='ratio between clipping bound and std of noise applied to gradients (default: 1.1)')
    params = vars(parser.parse_args())

    generator = train(params)

    with open('dp_generator.dat', 'wb') as f:
        torch.save(generator, f)

