import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ops import reparameterize
from ctgan.data_transformer import DataTransformer
from ctgan.data_sampler import DataSampler


class CNNEncoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(32 * input_dim, latent_dim)
        self.fc_logvar = nn.Linear(32 * input_dim, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class Residual(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([x, out], dim=1)


class Generator(nn.Module):
    def __init__(self, input_dim, dims, data_dim):
        super().__init__()
        seq = []
        for dim in dims:
            seq.append(Residual(input_dim, dim))
            input_dim += dim
        seq.append(nn.Linear(input_dim, data_dim))
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, dims, pac=10):
        super().__init__()
        self.pac = pac
        total_in = input_dim * pac
        seq = []
        for dim in dims:
            seq += [nn.Linear(total_in, dim), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            total_in = dim
        seq.append(nn.Linear(total_in, 1))
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        assert x.size(0) % self.pac == 0
        return self.model(x.view(x.size(0) // self.pac, -1))


class VAE_CTGAN(nn.Module):
    def __init__(self, embedding_dim, z_dim, device, batch_size,
                 lr_g, lr_d, lr_e, discriminator_steps, pac,
                 sample_dir, checkpoint_dir, log_frequency):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.pac = pac
        self.sample_dir = sample_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_frequency = log_frequency
        self.d_steps = discriminator_steps

        self.encoder = CNNEncoder(input_dim=z_dim, latent_dim=z_dim).to(device)
        self.generator = None
        self.discriminator = None

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
        import pandas as pd
        data = pd.read_csv(train_path)
        discrete_cols = data.select_dtypes(include='object').columns.tolist()

        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_cols)
        train_data = self._transformer.transform(data)
        self._data_sampler = DataSampler(train_data, self._transformer.output_info_list, self.log_frequency)

        cond_dim = self._data_sampler.dim_cond_vec()
        data_dim = self._transformer.output_dimensions

        self.generator = Generator(self.z_dim + cond_dim, [256, 256], data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim + cond_dim, [256, 256], self.pac).to(self.device)

        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        mean = torch.zeros(self.batch_size, self.z_dim).to(self.device)
        std = mean + 1

        for epoch in range(epochs):
            steps = max(len(train_data) // self.batch_size, 1)
            for _ in range(steps):
                for _ in range(self.d_steps):
                    z_noise = torch.normal(mean=mean, std=std)
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

                # Train Generator + Encoder
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
