import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        conv_out_dim = 64 * input_dim  # 입력을 (B, 1, input_dim)으로 받는다고 가정

        self.fc_mu = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        h = self.conv(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, input_dim, gside, num_channels):
        super().__init__()
        self.init_dim = (num_channels * 8, gside // 8, gside // 8)
        self.fc = nn.Linear(input_dim, int(np.prod(self.init_dim)))

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(num_channels * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels * 8, num_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels * 4, num_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels * 2, 1, 4, 2, 1)  # (B, 1, gside, gside)
        )

    def forward(self, z):
        x = self.fc(z).view(-1, *self.init_dim)
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self, dside, num_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, num_channels, 4, 2, 1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels * 2, num_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels * 4, 1, dside // 8, 1, 0)
        )

    def forward(self, x):
        h = self.conv[:-1](x)
        out = self.conv[-1](h)
        return out.view(-1, 1), h.view(h.size(0), -1) 

class Classifier(nn.Module):
    def __init__(self, dside, num_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, num_channels, 4, 2, 1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels * 2, num_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(num_channels * 4),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear((num_channels * 4) * (dside // 8) * (dside // 8), num_classes)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class VAE_CTABGAN(nn.Module):
    def __init__(self, embedding_dim, z_dim, device, batch_size, lr, sample_dir, checkpoint_dir):
        super(VAE_CTABGAN, self).__init__()
        self.embedding_dim = embedding_dim
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.sample_dir = sample_dir
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, encoder, generator, discriminator, epoch):
        save_path = f"{self.checkpoint_dir}/vae_ctabgan_epoch{epoch}.pth"
        torch.save({
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, save_path)
        print(f"✅ Checkpoint saved to {save_path}")
