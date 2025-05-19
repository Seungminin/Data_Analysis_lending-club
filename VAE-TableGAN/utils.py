import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
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
            nn.BatchNorm2d(feature_maps * 8), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps,   4, 2, 1),
            nn.BatchNorm2d(feature_maps),       nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps,     output_channels,4, 2, 1),
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
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps*4), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.main(x))

class Classifier(nn.Module):
    def __init__(self, input_channels=1, feature_maps=64, num_classes=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps), nn.ReLU(True),
            nn.Conv2d(feature_maps, feature_maps*2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps*2), nn.ReLU(True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps*4), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(feature_maps*4, num_classes)

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        return self.classifier(x)

class VAETableGan(nn.Module):
    def __init__(self,
                 input_dim,
                 batch_size,
                 y_dim,
                 delta_mean,
                 delta_var,
                 attrib_num,
                 label_col,
                 checkpoint_dir,
                 sample_dir,
                 dataset_name,
                 test_id,
                 device,
                 lambda_vae: float = 1.0,
                 lambda_info: float = 1.0,
                 lambda_advcls: float = 1.0,
                 val_split: float = 0.1):
        super().__init__()
        self.latent_dim     = 32
        self.batch_size     = batch_size
        self.delta_mean     = delta_mean
        self.delta_var      = delta_var
        self.attrib_num     = attrib_num
        self.label_col      = label_col
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir     = sample_dir
        self.dataset_name   = dataset_name
        self.test_id        = test_id
        self.device         = device
        self.input_dim      = input_dim

        self.lambda_vae     = lambda_vae
        self.lambda_info    = lambda_info
        self.lambda_advcls  = lambda_advcls
        self.val_split      = val_split

        self.encoder       = Encoder(input_dim*input_dim, self.latent_dim).to(device)
        self.generator     = Generator(z_dim=self.latent_dim, output_channels=1).to(device)
        self.discriminator = Discriminator(input_channels=1).to(device)
        self.classifier    = Classifier(input_channels=1, num_classes=y_dim).to(device)

        self.opt_enc  = torch.optim.Adam(self.encoder.parameters(), lr=2e-4)
        self.opt_gen  = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)
        self.opt_cls  = torch.optim.Adam(self.classifier.parameters(), lr=2e-4)

        self.prev_gmean = None
        self.prev_gvar  = None

    def forward(self, x):
        z, mu, logvar = self.encoder(x.view(x.size(0), -1))
        return self.generator(z), mu, logvar

    def compute_losses(self, real_data, labels):
        real_flat = real_data.view(real_data.size(0), -1)
        z, mu, logvar = self.encoder(real_flat)
        fake_data     = self.generator(z)

        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data.detach())

        # VAE
        rec_loss = F.mse_loss(fake_data, real_data)
        kl_loss  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = rec_loss + kl_loss

        # adversarial + classification
        adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
        real_logits = self.classifier(real_data)
        fake_logits = self.classifier(fake_data)
        cls_loss    = F.cross_entropy(real_logits, labels) + F.cross_entropy(fake_logits, labels)
        advcls_loss = adv_loss + cls_loss

        # info matching
        mean_r = torch.mean(real_score, dim=0, keepdim=True)
        mean_f = torch.mean(fake_score, dim=0, keepdim=True)
        var_r  = torch.var(real_score,  dim=0, keepdim=True)
        var_f  = torch.var(fake_score,  dim=0, keepdim=True)
        if self.prev_gmean is None:
            self.prev_gmean = mean_r.detach()
            self.prev_gvar  = var_r.detach()
        gmean_r = 0.99 * self.prev_gmean + 0.01 * mean_r
        gmean_f = 0.99 * self.prev_gmean + 0.01 * mean_f
        gvar_r  = 0.99 * self.prev_gvar  + 0.01 * var_r
        gvar_f  = 0.99 * self.prev_gvar  + 0.01 * var_f
        info_loss = (
            torch.sum(torch.clamp(torch.abs(gmean_r - gmean_f) - self.delta_mean, min=0.0)) +
            torch.sum(torch.clamp(torch.abs(gvar_r  - gvar_f)  - self.delta_var,  min=0.0))
        )
        self.prev_gmean = gmean_r.detach()
        self.prev_gvar  = gvar_r.detach()

        g_loss = (
            self.lambda_vae    * vae_loss   +
            self.lambda_info   * info_loss  +
            self.lambda_advcls * advcls_loss
        )
        d_loss = (
            F.binary_cross_entropy(real_score, torch.ones_like(real_score)) +
            F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
        )
        return g_loss, d_loss, cls_loss.item(), rec_loss.item(), kl_loss.item(), adv_loss.item(), info_loss.item()

    def train_model(self, args):
        full_ds   = TabularDataset(f"dataset/{self.dataset_name}/{self.dataset_name}.csv",
                                   self.input_dim, self.attrib_num, self.label_col)
        n_total   = len(full_ds)
        n_val     = int(self.val_split * n_total)
        n_train   = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        mlflow.set_experiment("VAE-TableGAN")
        mlflow.start_run()
        mlflow.log_param("epochs",        args.epoch)
        mlflow.log_param("lambda_vae",    self.lambda_vae)
        mlflow.log_param("lambda_info",   self.lambda_info)
        mlflow.log_param("lambda_advcls", self.lambda_advcls)

        for epoch in tqdm(range(args.epoch), desc="VAE-TableGAN"):
            self.train()
            for real_data, labels in train_loader:
                real_data, labels = real_data.to(self.device), labels.to(self.device)
                self.opt_enc.zero_grad()
                self.opt_gen.zero_grad()
                self.opt_disc.zero_grad()
                self.opt_cls.zero_grad()

                g_loss, d_loss, c_loss, r_loss, kl_loss, adv_loss, info_loss = \
                    self.compute_losses(real_data, labels)

                g_loss.backward(retain_graph=True)
                self.opt_enc.step()
                self.opt_gen.step()

                d_loss.backward()
                self.opt_disc.step()

                self.opt_cls.zero_grad()
                real_logits = self.classifier(real_data)
                cls_only   = F.cross_entropy(real_logits, labels)
                cls_only.backward()
                self.opt_cls.step()

            # validation
            self.eval()
            with torch.no_grad():
                real_list = [x for x, _ in val_loader]
                real_batch = torch.cat(real_list, 0).to(self.device)
                n_val_total = real_batch.size(0)
                z = torch.randn(n_val_total, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)

                real_flat = real_batch.view(n_val_total, -1)
                fake_flat = fake_batch.view(n_val_total, -1)

                mmd = compute_mmd(real_flat, fake_flat).item()
                wd  = compute_wasserstein(real_flat.cpu().numpy(),
                                          fake_flat.cpu().numpy())

            mlflow.log_metric("val_mmd", mmd, step=epoch)
            mlflow.log_metric("val_wd",  wd,  step=epoch)
            wandb.log({"val_mmd": mmd, "val_wd": wd}, step=epoch)

            mlflow.log_metric("g_loss",     g_loss.item(), step=epoch)
            mlflow.log_metric("d_loss",     d_loss.item(), step=epoch)
            mlflow.log_metric("c_loss",     c_loss,       step=epoch)
            mlflow.log_metric("recon_loss", r_loss,       step=epoch)
            mlflow.log_metric("kl_loss",    kl_loss,      step=epoch)
            mlflow.log_metric("adv_loss",   adv_loss,     step=epoch)
            mlflow.log_metric("info_loss",  info_loss,    step=epoch)

            wandb.log({
                "g_loss":     g_loss.item(),
                "d_loss":     d_loss.item(),
                "c_loss":     c_loss,
                "recon_loss": r_loss,
                "kl_loss":    kl_loss,
                "adv_loss":   adv_loss,
                "info_loss":  info_loss,
            }, step=epoch)

            print(f"[{epoch+1}/{args.epoch}] G={g_loss.item():.4f}, D={d_loss.item():.4f}, "
                  f"C={c_loss:.4f}, Recon={r_loss:.4f}, KL={kl_loss:.4f}, INFO={info_loss:.4f}, "
                  f"MMD={mmd:.4f}, WD={wd:.4f}")

            self.train()

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt'))
        mlflow.end_run()
        wandb.finish()
