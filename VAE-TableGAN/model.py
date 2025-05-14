# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from ops import reparameterize, init_weights

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
        batch_size = real_data.size(0)
        z, mu, logvar = self.encoder(real_data)
        fake_data = self.generator(z)

        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data.detach())

        recon_loss = F.mse_loss(fake_data, real_data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
        disc_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                    F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))

        # classifier
        real_logits = self.classifier(real_data)
        fake_logits = self.classifier(fake_data)
        class_loss = F.cross_entropy(real_logits, labels) + F.cross_entropy(fake_logits, labels)

        # info loss
        D_features = real_score
        D_features_ = fake_score

        mean_real = torch.mean(D_features, dim=0, keepdim=True)
        mean_fake = torch.mean(D_features_, dim=0, keepdim=True)
        var_real = torch.var(D_features, dim=0, keepdim=True)
        var_fake = torch.var(D_features_, dim=0, keepdim=True)

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

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt'))

    def load(self):
        path = os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt')
        self.load_state_dict(torch.load(path))
        self.eval()