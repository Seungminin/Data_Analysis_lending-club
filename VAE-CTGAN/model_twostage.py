import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import wandb

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from ops import reparameterize
from data_transformer import DataTransformer
from data_sampler import DataSampler

class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=64, input_length=209):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.latent_dim = latent_dim
        self.input_length = input_length

        # output: 64 channels × input_length
        conv_output_dim = 64 * input_length

        self.fc_mu = nn.Sequential(
            nn.Linear(conv_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(conv_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, input_length]
        h = self.flatten(self.conv(x))  # [B, 64 × input_length]
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
        self.embedding_dim = embedding_dim
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size
        self.pac = pac
        self.sample_dir = sample_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_frequency = log_frequency
        self.d_steps = discriminator_steps

        self.encoder = CNNEncoder(latent_dim=z_dim, input_length=209).to(device)
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
        assert real_data.size(0) == fake_data.size(0)
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data)

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

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def train_from_transformed(self, data_path: str, transformer_path: str, epochs: int, warmup_epochs: int = 40,
                                alpha_adv: float = 1.0, alpha_rec: float = 1.0, alpha_kl: float = 1.0):
        from utils import load_transformer

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._transformer = load_transformer(transformer_path)
        data_transformed = pd.read_csv(data_path).values.astype(np.float32)
        self._data_sampler = DataSampler(data_transformed, self._transformer.output_info_list, self.log_frequency)

        cond_dim = self._data_sampler.dim_cond_vec()
        data_dim = self._transformer.output_dimensions

        self.generator = Generator(self.z_dim + cond_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim + cond_dim, self.pac).to(self.device)

        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        data_tensor = torch.tensor(data_transformed, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Epoch"):
            batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

            total_d_loss = total_g_loss = total_adv_loss = total_rec_loss = total_kl_loss = 0.0
            num_batches = 0

            for real_batch, in batch_iter:
                real_batch = real_batch.to(self.device)
                condvec = self._data_sampler.sample_condvec(real_batch.size(0))

                if condvec is None:
                    print("⚠️ Skipped batch due to missing condvec.")
                    continue
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device).float()

                z_latent, mu, logvar = self.encoder(real_batch)
                zc = torch.cat([z_latent, c1], dim=1)
                fake = self.generator(zc)

                min_len = min(real_batch.size(0), fake.size(0), c1.size(0))
                min_len = (min_len // self.pac) * self.pac
                if min_len == 0:
                    continue

                real = real_batch[:min_len]
                fake = fake[:min_len]
                c1 = c1[:min_len]

                x_fake = torch.cat([fake, c1], dim=1)
                x_real = torch.cat([real, c1], dim=1)

                rec_loss = F.mse_loss(fake, real)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                if epoch < warmup_epochs:
                    # VAE stage
                    g_loss = alpha_rec * rec_loss + alpha_kl * kl
                    self.opt_g.zero_grad()
                    self.opt_e.zero_grad()
                    g_loss.backward()
                    self.opt_g.step()
                    self.opt_e.step()

                    total_g_loss += g_loss.item()
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl.item()

                else:
                    # Full adversarial stage
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
                    g_loss = alpha_adv * adv_loss + alpha_rec * rec_loss + alpha_kl * kl

                    self.opt_g.zero_grad()
                    self.opt_e.zero_grad()
                    g_loss.backward()
                    self.opt_g.step()
                    self.opt_e.step()

                    total_d_loss += d_loss.item()
                    total_g_loss += g_loss.item()
                    total_adv_loss += adv_loss.item()
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl.item()

                num_batches += 1

            if num_batches > 0:
                wandb.log({
                    "epoch": epoch,
                    "D_loss": total_d_loss / num_batches,
                    "G_loss": total_g_loss / num_batches,
                    "Adv_loss": total_adv_loss / num_batches,
                    "Rec_loss": total_rec_loss / num_batches,
                    "KL_loss": total_kl_loss / num_batches
                })

            if epoch > 200 and (epoch % 20 == 0 or epoch == epochs - 1):
                torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,f"vae_ctgan_{epoch}epoch.pt"))


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

    def load(self, path, transformer_path, data_path):
        from utils import load_transformer

        self._transformer = load_transformer(transformer_path)
        data = pd.read_csv(data_path).values.astype(np.float32)
        self._data_sampler = DataSampler(data, self._transformer.output_info_list, self.log_frequency)

        cond_dim = self._data_sampler.dim_cond_vec()
        data_dim = self._transformer.output_dimensions

        print(f"cond_dim : {cond_dim}, data_dim : {data_dim}, pac : {self.pac}\n")
        self.generator = Generator(self.z_dim + cond_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim + cond_dim, self.pac).to(self.device)

        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()
        print(f"✅ Loaded checkpoint from: {path}")
