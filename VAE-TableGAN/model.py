import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from ops import reparameterize, init_weights
from utils import padding_duplicating, reshape
import mlflow

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.apply(init_weights)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class VAETableGan(nn.Module):
    def __init__(self, input_dim, batch_size, y_dim, alpha, beta, delta_mean, delta_var,
                 attrib_num, label_col, checkpoint_dir, sample_dir, dataset_name, test_id, device):
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

        self.encoder = Encoder(input_dim, self.latent_dim).to(device)
        self.generator = Generator(self.latent_dim, input_dim).to(device)
        self.discriminator = Discriminator(input_dim).to(device)
        self.classifier = Classifier(input_dim).to(device)

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=0.0002)
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.opt_cls = torch.optim.Adam(self.classifier.parameters(), lr=0.0002)

        self.prev_gmean = None
        self.prev_gvar = None

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.generator(z)
        return x_hat, mu, logvar

    def compute_losses(self, real_data, labels):
        z, mu, logvar = self.encoder(real_data)
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

    # model.py 내부 load_dataset 함수
    def load_dataset(self):
        data_path = f"dataset/{self.dataset_name}/{self.dataset_name}.csv"
        X = pd.read_csv(data_path)
        y = X['loan_status']
        X = X.drop(columns=['loan_status']) 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = pd.DataFrame(scaler.fit_transform(X))

        reshaped = reshape(X_scaled)

        X_tensor = torch.tensor(reshaped, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        return TensorDataset(X_tensor, y_tensor)


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

        for epoch in range(args.epoch):
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

            print(f"Epoch {epoch+1}/{args.epoch}: G={g_loss.item():.4f}, D={d_loss.item():.4f}, C={c_loss:.4f}, Recon={r_loss:.4f}, KL={kl_loss:.4f}, Info={info:.4f}")

        self.save()
        mlflow.end_run()
