import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):
    def __init__(self, example_dim, compression_dim, binary=True, device='cpu'):
        super(Autoencoder, self).__init__()

        self.compression_dim = compression_dim

        self.encoder = nn.Sequential(
            nn.Linear(example_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, compression_dim),
            nn.Tanh() if binary else nn.LeakyReLU(0.2)
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, example_dim),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_compression_dim(self):
        return self.compression_dim

