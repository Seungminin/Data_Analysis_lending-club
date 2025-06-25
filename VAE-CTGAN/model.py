import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import wandb

from ops import reparameterize
from data_transformer import DataTransformer
from data_sampler import DataSampler


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.latent_dim = latent_dim
        self.fc_mu = None
        self.fc_logvar = None

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.flatten(self.conv(x))
        if self.fc_mu is None:
            dim = h.shape[1]
            self.fc_mu = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, self.latent_dim)
            ).to(h.device)
            self.fc_logvar = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, self.latent_dim)
            ).to(h.device)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, input_dim, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, data_dim)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, pac=10):
        super().__init__()
        self.pac = pac
        total_in = input_dim * pac
        self.model = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        assert x.size(0) % self.pac == 0
        return self.model(x.view(x.size(0) // self.pac, -1))


class VAE_CTGAN(nn.Module):
    def __init__(self, embedding_dim, z_dim, device, batch_size,
                 lr_g, lr_d, lr_e, discriminator_steps, pac,
                 sample_dir, checkpoint_dir, log_frequency):
        super().__init__()
        self.embedding_dim  = embedding_dim
        self.z_dim          = z_dim
        self.device         = device
        self.batch_size     = batch_size
        self.pac            = pac
        self.sample_dir     = sample_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_frequency  = log_frequency
        self.d_steps        = discriminator_steps

        self.encoder        = CNNEncoder(latent_dim=z_dim).to(device)
        self.generator      = None
        self.discriminator  = None

        self.opt_g = None
        self.opt_d = None
        self.opt_e = optim.Adam(self.encoder.parameters(), lr=lr_e)

        self.lr_g = lr_g
        self.lr_d = lr_d

        self._transformer = None
        self._data_sampler = None

    def calc_gradient_penalty(self, real_data, fake_data, discriminator):
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=self.device)
        alpha = alpha.repeat(1, self.pac, real_data.size(1)).view(-1, real_data.size(1))
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        d_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def fit(self, train_path, epochs):
        data = pd.read_csv(train_path)
        discrete_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        print("Before DataTransformer")
        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_cols)
        train_data = self._transformer.transform(data)
        self._data_sampler = DataSampler(train_data, self._transformer.output_info_list, self.log_frequency)

        cond_dim = self._data_sampler.dim_cond_vec()
        data_dim = self._transformer.output_dimensions

        print("After DataTransformer")
        self.generator = Generator(self.z_dim + cond_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim + cond_dim, self.pac).to(self.device)

        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        mean = torch.zeros(self.batch_size, self.z_dim).to(self.device)
        std = mean + 1

        for epoch in range(epochs):
            steps = max(len(train_data) // self.batch_size, 1)
            for step in range(steps):
                for _ in range(self.d_steps):
                    condvec = self._data_sampler.sample_condvec(self.batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device).float()
                        real = self._data_sampler.sample_data(train_data, self.batch_size, col, opt)
                        real = torch.from_numpy(real.astype('float32')).to(self.device)
                        z_latent, mu, logvar = self.encoder(real)
                        zc = torch.cat([z_latent, c1], dim=1)
                        fake = self.generator(zc)

                        x_fake = torch.cat([fake, c1], dim=1)
                        x_real = torch.cat([real, c1], dim=1)
                        gp = self.calc_gradient_penalty(x_real, x_fake, self.discriminator)
                        d_real = self.discriminator(x_real)
                        d_fake = self.discriminator(x_fake)
                        d_loss = -(torch.mean(d_real) - torch.mean(d_fake)) + gp

                        self.opt_d.zero_grad()
                        d_loss.backward()
                        self.opt_d.step()

                z_latent, mu, logvar = self.encoder(real)
                zc = torch.cat([z_latent, c1], dim=1)
                fake = self.generator(zc)
                x_fake = torch.cat([fake, c1], dim=1)

                g_score = self.discriminator(x_fake)
                adv_loss = -torch.mean(g_score)
                rec_loss = F.mse_loss(fake, real)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                g_loss = adv_loss + rec_loss + kl

                self.opt_g.zero_grad()
                self.opt_e.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                self.opt_e.step()

                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "D_loss": d_loss.item(),
                    "G_loss": g_loss.item(),
                    "Adv_loss": adv_loss.item(),
                    "Rec_loss": rec_loss.item(),
                    "KL_loss": kl.item()
                })

    def sample(self, n):
        steps = n // self.batch_size + 1
        out = []
        for _ in range(steps):
            z = torch.randn(self.batch_size, self.z_dim).to(self.device)
            cond = self._data_sampler.sample_original_condvec(self.batch_size)
            if cond is not None:
                cond = torch.from_numpy(cond).to(self.device).float()
                zc = torch.cat([z, cond], dim=1)
            else:
                zc = z
            fake = self.generator(zc)
            act = fake.detach().cpu().numpy()
            out.append(act)
        samples = np.concatenate(out, axis=0)[:n]
        return self._transformer.inverse_transform(samples)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()