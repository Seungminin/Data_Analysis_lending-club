import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from ops import reparameterize, init_weights
from utils import TabularDataset
import mlflow

class Encoder(nn.Module):
    def __init__(self, input_dim=32*32, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

class Generator(nn.Module):
    def __init__(self, z_dim=32, output_shape=(1, 64, 64)):
        super().__init__()
        self.init_size = output_shape[1] // 4
        self.fc = nn.Linear(z_dim, 256 * self.init_size * self.init_size)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),  # Remove inplace=True
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # Remove inplace=True
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # Remove inplace=True
            nn.Conv2d(64, output_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 256, self.init_size, self.init_size)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # Remove inplace=True
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  # Remove inplace=True
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),  # Remove inplace=True
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        feat = self.conv(x)
        out = self.classifier(feat)
        if return_features:
            return out, feat.view(x.size(0), -1)
        return out

class Classifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),  # Remove inplace=True
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),  # Remove inplace=True
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),  # Remove inplace=True
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.conv(x)
        return self.fc(feat.view(x.size(0), -1))

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
                 lr,
                 pre_epochs,
                 epochs,
                 lambda_vae: float = 1.0,
                 lambda_info: float = 1.0,
                 lambda_advcls: float = 1.0):
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
        self.input_dim      = input_dim * input_dim
        self.pre_epochs     = pre_epochs
        self.epochs         = epochs
        self.global_step    = 0
        self.lambda_vae     = lambda_vae
        self.lambda_info    = lambda_info
        self.lambda_advcls  = lambda_advcls

        # modules
        self.encoder       = Encoder(self.input_dim, self.latent_dim).to(device)
        self.generator     = Generator(z_dim=self.latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.classifier    = Classifier(num_classes=y_dim).to(device)

        # optimizers
        self.opt_enc  = torch.optim.Adam(self.encoder.parameters(),       lr=lr)
        self.opt_gen  = torch.optim.Adam(list(self.generator.parameters()),     lr=lr)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr*0.1)
        self.opt_cls  = torch.optim.Adam(self.classifier.parameters(),    lr=lr)

        self.prev_gmean = None
        self.prev_gvar  = None

    def compute_gan_losses(self, real_data, fake_data, labels):
        real_score = self.discriminator(real_data)
        fake_score = self.discriminator(fake_data.detach())
        adv_loss   = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
        d_loss     = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                     F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))

        real_logits= self.classifier(real_data)
        fake_logits= self.classifier(fake_data)
        cls_loss   = F.cross_entropy(real_logits, labels) + F.cross_entropy(fake_logits, labels)
        advcls     = adv_loss + cls_loss
        return advcls, d_loss, adv_loss, cls_loss

    def compute_info_loss(self, real_feat, fake_feat):
        mean_r = real_feat.mean(0, keepdim=True)
        mean_f = fake_feat.mean(0, keepdim=True)
        var_r  = real_feat.var(0, keepdim=True)
        var_f  = fake_feat.var(0, keepdim=True)

        if self.prev_gmean is None:
            self.prev_gmean = mean_r.detach()
            self.prev_gvar  = var_r.detach()

        gmean_r = 0.99*self.prev_gmean + 0.01*mean_r
        gmean_f = 0.99*self.prev_gmean + 0.01*mean_f
        gvar_r  = 0.99*self.prev_gvar  + 0.01*var_r
        gvar_f  = 0.99*self.prev_gvar  + 0.01*var_f

        info_loss = (
            torch.clamp((gmean_r - gmean_f).abs() - self.delta_mean, min=0).sum() +
            torch.clamp((gvar_r  - gvar_f ).abs() - self.delta_var,  min=0).sum()
        )

        self.prev_gmean, self.prev_gvar = gmean_r.detach(), gvar_r.detach()
        return info_loss

    def save(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_{self.test_id}_{self.pre_epochs}_model.pt")
        torch.save(self.state_dict(), path)

    def load(self):
        path = os.path.join(self.checkpoint_dir, f"{self.dataset_name}_{self.test_id}_{self.pre_epochs}_model.pt")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

    def train_model(self, args):
        ds = TabularDataset(f"dataset/{self.dataset_name}/{self.dataset_name}.csv",
                            int((self.input_dim)**0.5), self.attrib_num, self.label_col)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        mlflow.set_experiment("VAE-TableGAN")
        mlflow.start_run()
        wandb.init(project="vae-tablegan", name=self.test_id, config=vars(args))

        for epoch in tqdm(range(self.epochs), desc="One-Stage Training"):
            g_totals = {'vae': 0, 'info': 0, 'advcls': 0}
            d_total = 0

            torch.autograd.set_detect_anomaly(True)
            
            for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                flat = x.view(x.size(0), -1)

                # === Update Discriminator ===
                self.discriminator.train()
                self.opt_disc.zero_grad()
                
                z, mu, logvar = self.encoder(flat)
                x_hat = self.generator(z)
                
                real_score = self.discriminator(x)
                fake_score = self.discriminator(x_hat.detach())
                d_loss = (F.binary_cross_entropy(real_score, torch.ones_like(real_score)) +
                         F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score)))
                
                d_loss.backward()
                self.opt_disc.step()

                # === Update Generator and Encoder ===
                self.generator.train()
                self.encoder.train()
                self.opt_gen.zero_grad()
                self.opt_enc.zero_grad()

                # Fresh forward pass
                z, mu, logvar = self.encoder(flat)
                x_hat = self.generator(z)

                # VAE losses
                rec = F.mse_loss(x_hat, x)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = rec + kl

                # GAN losses
                fake_score = self.discriminator(x_hat)
                adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
                
                # Classification losses
                real_logits = self.classifier(x)
                fake_logits = self.classifier(x_hat)
                cls_loss = F.cross_entropy(real_logits, y) + F.cross_entropy(fake_logits, y)
                
                # Info loss
                _, real_feat = self.discriminator(x, return_features=True)
                _, fake_feat = self.discriminator(x_hat, return_features=True)
                info_loss = self.compute_info_loss(real_feat, fake_feat)

                # Combined generator loss
                g_loss = (self.lambda_vae * vae_loss +
                         self.lambda_info * info_loss +
                         self.lambda_advcls * (adv_loss + cls_loss))

                g_loss.backward()
                self.opt_gen.step()
                self.opt_enc.step()

                # === Second Generator Update ===
                self.opt_gen.zero_grad()
                
                # Use detached z for second update
                with torch.no_grad():
                    z_detached, _, _ = self.encoder(flat)
                
                x_hat_2 = self.generator(z_detached)
                rec_2 = F.mse_loss(x_hat_2, x)
                fake_score_2 = self.discriminator(x_hat_2)
                adv_loss_2 = F.binary_cross_entropy(fake_score_2, torch.ones_like(fake_score_2))
                
                fake_logits_2 = self.classifier(x_hat_2)
                cls_loss_2 = F.cross_entropy(fake_logits_2, y)
                
                _, fake_feat_2 = self.discriminator(x_hat_2, return_features=True)
                info_loss_2 = self.compute_info_loss(real_feat.detach(), fake_feat_2)

                g_loss_2 = (self.lambda_vae * rec_2 +
                           self.lambda_info * info_loss_2 +
                           self.lambda_advcls * (adv_loss_2 + cls_loss_2))

                g_loss_2.backward()
                self.opt_gen.step()

                # === Update Classifier ===
                self.classifier.train()
                self.opt_cls.zero_grad()
                real_logits_cls = self.classifier(x)
                cls_real_loss = F.cross_entropy(real_logits_cls, y)
                cls_real_loss.backward()
                self.opt_cls.step()

                # === Accumulate loss for logging ===
                g_totals['vae'] += vae_loss.item()
                g_totals['info'] += info_loss.item()
                g_totals['advcls'] += (adv_loss + cls_loss).item()
                d_total += d_loss.item()

            wandb.log({
                "vae_loss": g_totals['vae'] / len(loader),
                "info_loss": g_totals['info'] / len(loader),
                "advcls_loss": g_totals['advcls'] / len(loader),
                "d_loss": d_total / len(loader)
            }, step=self.global_step)

            self.global_step += 1

        self.save()
        mlflow.end_run()
        wandb.finish()