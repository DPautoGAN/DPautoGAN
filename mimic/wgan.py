import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
import mimic_dataset


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, binary, device='cpu'):
        super(Generator, self).__init__()
        self.w1 = nn.Linear(input_dim, 128, bias=False).to(device)
        self.bn1 = nn.BatchNorm1d(128, 0.01).to(device)
        self.a1 = nn.ReLU().to(device)

        self.w2 = nn.Linear(128, 128, bias=False).to(device)
        self.bn2 = nn.BatchNorm1d(128, 0.01).to(device)
        self.a2 = nn.ReLU().to(device)

        self.w3 = nn.Linear(128, output_dim, bias=False).to(device)
        self.bn3 = nn.BatchNorm1d(output_dim, 0.01).to(device)
        self.a3 = (nn.Tanh() if binary else nn.ReLU()).to(device)

    def forward(self, x):
        x = self.a1(self.bn1(self.w1(x))) + x
        x = self.a2(self.bn2(self.w2(x))) + x
        x = self.a3(self.bn3(self.w3(x))) + x
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, device='cpu'):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.model(torch.cat([x, x.mean(dim=0).repeat(x.size(0), 1)], dim=1))


def train(params):
    dataset = {
        'mimic': mimic_dataset,
    }[params['dataset']]

    train_dataset, _, _, _ = dataset.get_datasets()

    dataloader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
    )

    with open('autoencoder.dat', 'rb') as f:
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
        input_dim=np.prod(next(iter(dataloader))[0].shape),
        device=params['device'],
    )

    d_optimizer = torch.optim.RMSprop(
        params=discriminator.parameters(),
        lr=params['lr'],
        alpha=params['alpha'],
        weight_decay=params['l2_penalty'],
    )

    batches_done = 0
    for epoch in range(params['epochs']):
        for i, real in enumerate(dataloader):
            real = real.to(params['device'])

            z = torch.randn(real.size(0), params['latent_dim'], device=params['device'], requires_grad=False)
            fake = decoder(generator(z)).detach()
            d_optimizer.zero_grad()
            d_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
            d_loss.backward()
            d_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.data.clamp_(-params['clip_value'], params['clip_value'])

            if i % params['d_updates'] == 0:
                z = torch.randn(real.size(0), params['latent_dim'], device=params['device'], requires_grad=False)
                fake = decoder(generator(z))
                g_optimizer.zero_grad()
                g_loss = -torch.mean(discriminator(fake))
                g_loss.backward()
                g_optimizer.step()

            if batches_done % 10 == 0:
                print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (
                    epoch, params['epochs'], i, len(dataloader), d_loss.item(), g_loss.item())
                )
            batches_done += 1

        if epoch % 20 == 0:
            with open('wgans/{}.dat'.format(str(epoch)), 'wb') as f:
                torch.save(generator, f)

    return generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.99, help='alpha learning parameter (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=1000, help='input batch size for training (default: 1000)')
    parser.add_argument('--clip-value', type=float, default=0.01, help='upper bound on weights of the discriminator (default: 0.01)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='whether or not to use cuda (default: cuda if available)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 1000)')
    parser.add_argument('--latent-dim', type=int, default=128, help='dimensionality of the latent space (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
    parser.add_argument('--binary', type=bool, default=True, help='whether data type is binary (default: true)')
    parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    parser.add_argument('--dataset', type=str, default='mimic', help='the dataset to be used for training (default: mimic)')
    parser.add_argument('--d-updates', type=int, default=2, help='number of iterations to update discriminator per generator update (default: 2)')
    params = vars(parser.parse_args())

    generator = train(params)

    with open('w_generator.dat', 'wb') as f:
        torch.save(generator, f)
