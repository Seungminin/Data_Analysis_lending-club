import os

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from ops import reparameterize, init_weights
from utils import TabularDataset, compute_mmd, compute_wasserstein
import mlflow

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.apply(init_weights)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Generator(nn.Module):
    def __init__(self, z_dim=32, feature_maps=64, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(z_dim, feature_maps * 8 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),      # 16→32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, output_channels, 4, 2, 1),       # 32→64 ✅
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return self.classifier(x)

class Classifier(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64, num_classes=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(feature_maps * 4, num_classes)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class VAETableGan(nn.Module):
    def __init__(self, input_dim, batch_size, y_dim, alpha, beta, delta_mean, delta_var,
                 attrib_num, label_col, checkpoint_dir, sample_dir, dataset_name, test_id, device, lr, epochs):
        super().__init__()
        self.latent_dim = 32
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.delta_mean = delta_mean
        self.delta_var = delta_var
        self.label_col = label_col
        self.attrib_num = attrib_num
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.dataset_name = dataset_name
        self.test_id = test_id
        self.device = device
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs

        D = input_dim * input_dim
        self.encoder = Encoder(D, self.latent_dim).to(device)
        self.generator = Generator(z_dim=self.latent_dim, output_channels=1).to(device)
        self.discriminator = Discriminator(input_channels=1).to(device)
        self.classifier = Classifier(input_channels=1).to(device)

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.opt_cls = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        self.prev_gmean = None
        self.prev_gvar = None

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.generator(z)
        return x_hat, mu, logvar

    def compute_losses(self, real_data, labels):
        real_data_flat = real_data.view(real_data.size(0), -1)
        z, mu, logvar = self.encoder(real_data_flat)
        fake_data = self.generator(z)

        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data.detach())

        recon_loss = F.mse_loss(fake_data, real_data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
        disc_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                    F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))

        real_logits = self.classifier(real_data)
        fake_logits = self.classifier(fake_data)
        class_loss = F.cross_entropy(real_logits, labels) + F.cross_entropy(fake_logits, labels)

        mean_real = torch.mean(real_score, dim=0, keepdim=True)
        mean_fake = torch.mean(fake_score, dim=0, keepdim=True)
        var_real = torch.var(real_score, dim=0, keepdim=True)
        var_fake = torch.var(fake_score, dim=0, keepdim=True)

        if self.prev_gmean is None:
            self.prev_gmean = mean_real.detach()
            self.prev_gvar = var_real.detach()

        gmean = 0.99 * self.prev_gmean + 0.01 * mean_real
        gmean_ = 0.99 * self.prev_gmean + 0.01 * mean_fake
        gvar = 0.99 * self.prev_gvar + 0.01 * var_real
        gvar_ = 0.99 * self.prev_gvar + 0.01 * var_fake

        info_loss = torch.sum(torch.clamp(torch.abs(gmean - gmean_) - self.delta_mean, min=0.0)) + \
                     torch.sum(torch.clamp(torch.abs(gvar - gvar_) - self.delta_var, min=0.0))

        self.prev_gmean = gmean.detach()
        self.prev_gvar = gvar.detach()

        g_loss = self.alpha * (adv_loss + class_loss) + self.beta * (recon_loss + kl_loss + info_loss)

        return g_loss, disc_loss, class_loss.item(), recon_loss.item(), kl_loss.item(), adv_loss.item(), info_loss.item()

    def load_dataset(self):
        return TabularDataset(
            csv_path=f"dataset/{self.dataset_name}/{self.dataset_name}.csv",
            input_dim=self.input_dim,
            attrib_num=self.attrib_num,
            label_col=self.label_col
        )

    def model_dir(self):
        return f"{self.dataset_name}_{self.batch_size}_{self.input_dim}_{self.test_id}"

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt'))

    def load(self):
        path = os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt')
        self.load_state_dict(torch.load(path))
        self.eval()

    def train_model(self, args):
        dataset = self.load_dataset()
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        mlflow.set_experiment("VAE-TableGAN")
        mlflow.start_run()
        mlflow.log_param("epoch", args.epoch)
        mlflow.log_param("alpha", self.alpha)
        mlflow.log_param("beta", self.beta)

        for epoch in tqdm(range(args.epoch), desc="VAE-TableGAN Time"):
            self.train()
            for real_data, labels in train_loader:
                real_data, labels = real_data.to(self.device), labels.to(self.device)

                self.opt_enc.zero_grad()
                self.opt_gen.zero_grad()
                self.opt_disc.zero_grad()
                self.opt_cls.zero_grad()

                g_loss, d_loss, c_loss, r_loss, kl_loss, adv_loss, info = self.compute_losses(real_data, labels)
                g_loss.backward(retain_graph=True)
                self.opt_enc.step()
                self.opt_gen.step()

                d_loss.backward()
                self.opt_disc.step()

                self.opt_cls.zero_grad()
                class_logits = self.classifier(real_data)
                loss_cls = F.cross_entropy(class_logits, labels)
                loss_cls.backward()
                self.opt_cls.step()

            mlflow.log_metric("g_loss", g_loss.item(), step=epoch)
            mlflow.log_metric("d_loss", d_loss.item(), step=epoch)
            mlflow.log_metric("c_loss", c_loss, step=epoch)
            mlflow.log_metric("recon_loss", r_loss, step=epoch)
            mlflow.log_metric("kl_loss", kl_loss, step=epoch)
            mlflow.log_metric("adv_loss", adv_loss, step=epoch)
            mlflow.log_metric("info_loss", info, step=epoch)

            # ✅ W&B 로깅
            wandb.log({
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "c_loss": c_loss,
                "recon_loss": r_loss,
                "kl_loss": kl_loss,
                "adv_loss": adv_loss,
                "info_loss": info,
            }, step=epoch)

            print(f"Epoch {epoch+1}/{args.epoch}: G={g_loss.item():.4f}, D={d_loss.item():.4f}, C={c_loss:.4f}, Recon={r_loss:.4f}, KL={kl_loss:.4f}, Info={info:.4f}")

        self.save()
        mlflow.end_run()
        wandb.finish()