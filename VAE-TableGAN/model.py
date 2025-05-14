# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from ops import reparameterize, init_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.apply(init_weights)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.apply(init_weights)

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
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

        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=0.0002)
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_hat = self.generator(z)
        return x_hat, mu, logvar

    def compute_losses(self, real_data):
        z, mu, logvar = self.encoder(real_data)
        fake_data = self.generator(z)

        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data.detach())

        recon_loss = F.mse_loss(fake_data, real_data)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
        disc_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                    F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))

        gen_loss = self.alpha * adv_loss + self.beta * (recon_loss + kl_loss)
        return gen_loss, disc_loss, recon_loss.item(), kl_loss.item(), adv_loss.item()

    def train(self, epochs, lr, experiment):
        for epoch in range(epochs):
            # 이 부분은 데이터 로딩 로직에 따라 수정되어야 함
            raise NotImplementedError("Add your DataLoader loop here.")

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt'))

    def load(self):
        path = os.path.join(self.checkpoint_dir, f'{self.test_id}_model.pt')
        self.load_state_dict(torch.load(path))
        self.eval()