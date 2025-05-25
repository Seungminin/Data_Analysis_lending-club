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

##모델의 input_dimention이 높을 때, 많은 feature수를 사용할 때. 
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, 1024)
        self.bn1      = nn.BatchNorm1d(1024)
        self.fc_mu    = nn.Linear(1024, latent_dim)
        self.fc_logvar= nn.Linear(1024, latent_dim)
        self.apply(init_weights)

    def forward(self, x):
        h      = F.relu(self.bn1(self.fc1(x)))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self, z_dim=32, feature_maps=32, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(z_dim, feature_maps*8*4*4)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(feature_maps*8), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, 4,2,1),
            nn.BatchNorm2d(feature_maps*4), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, 4,2,1),
            nn.BatchNorm2d(feature_maps*2), nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps*2, feature_maps,   4,2,1),
            nn.BatchNorm2d(feature_maps),       nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps,     output_channels,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=1, feature_maps=32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4,2,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps*2, 4,2,1),
            nn.BatchNorm2d(feature_maps*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4,2,1),
            nn.BatchNorm2d(feature_maps*4), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        scores = self.classifier(features)
        if return_features:
            return scores, features.view(x.size(0), -1)
        return scores


class Classifier(nn.Module):
    def __init__(self, input_channels=1, feature_maps=32, num_classes=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4,2,1),
            nn.BatchNorm2d(feature_maps), nn.ReLU(True),
            nn.Conv2d(feature_maps, feature_maps*2, 4,2,1),
            nn.BatchNorm2d(feature_maps*2), nn.ReLU(True),
            nn.Conv2d(feature_maps*2, feature_maps*4, 4,2,1),
            nn.BatchNorm2d(feature_maps*4), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(feature_maps*4, num_classes)

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        return self.classifier(x)
    

def fine_tune(self, train_loader):

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.opt_gen.param_groups[0]['lr'])

        for epoch in tqdm(range(self.epochs), desc = "Fine Tuning"):
            g_totals = {'vae':0, 'info':0, 'advcls':0}
            d_total = 0

            for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{self.epochs}"):
                x, y = x.to(self.device), y.to(self.device)
                flat = x.view(x.size(0), -1)
                with torch.no_grad(): 
                    z, mu, logvar = self.encoder(flat)

                x_hat = self.generator(z)

                rec = F.mse_loss(x_hat, x)
                kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae = rec + kl

                advcls, d_loss, adv, cls = self.compute_gan_losses(x, x_hat, y)

                real_score, real_feat = self.discriminator(x, return_features=True)
                fake_score, fake_feat = self.discriminator(x_hat, return_features=True)
                info = self.compute_info_loss(real_feat, fake_feat)

                g_loss = (self.lambda_vae * vae +
                          self.lambda_info * info +
                          self.lambda_advcls * advcls)

                # D update
                self.opt_disc.zero_grad()
                d_loss.backward()
                self.opt_disc.step()

                # G update twice
                for _ in range(2):
                    self.opt_gen.zero_grad()

                    x_hat = self.generator(z)
                    rec = F.mse_loss(x_hat, x)
                    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    vae = rec + kl

                    advcls, _, _, _ = self.compute_gan_losses(x, x_hat, y)
                    _, real_feat = self.discriminator(x, return_features=True)
                    _, fake_feat = self.discriminator(x_hat, return_features=True)
                    info = self.compute_info_loss(real_feat, fake_feat)

                    g_loss = (self.lambda_vae * vae +
                            self.lambda_info * info +
                            self.lambda_advcls * advcls)

                    g_loss.backward()
                    self.opt_gen.step()
                
                # C update
                self.opt_cls.zero_grad()
                real_logits = self.classifier(x)
                loss_rcls  = F.cross_entropy(real_logits, y)
                loss_rcls.backward()
                self.opt_cls.step()

                g_totals['vae']   += vae.item()
                g_totals['info']  += info.item()
                g_totals['advcls']+= advcls.item()
                d_total           += d_loss.item()

