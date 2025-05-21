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
            nn.Tanh()
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
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
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

    def pre_train(self, train_loader):
        for epoch in range(self.pre_epochs):
            rec_total, kl_total = 0, 0
            for x, _ in tqdm(train_loader, desc=f"VAE Pre-train Epoch {epoch+1}/{self.pre_epochs}"):
                x = x.to(self.device)
                flat = x.view(x.size(0), -1)
                z, mu, logvar = self.encoder(flat)
                x_hat = self.generator(z)
                rec = F.mse_loss(x_hat, x)
                kl  = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
                loss = rec + kl

                self.opt_enc.zero_grad(); self.opt_gen.zero_grad()
                loss.backward()
                self.opt_enc.step(); self.opt_gen.step()

                rec_total += rec.item(); kl_total += kl.item()

            wandb.log({
                "pre_rec_loss": rec_total/len(train_loader),
                "pre_kl_loss":  kl_total/len(train_loader)
            }, step=self.global_step)
            self.global_step += 1

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


            wandb.log({
                "ft_vae_loss":   g_totals['vae']/len(train_loader),
                "ft_info_loss":  g_totals['info']/len(train_loader),
                "ft_advcls_loss":g_totals['advcls']/len(train_loader),
                "ft_d_loss":     d_total/len(train_loader)
            }, step=self.global_step)
            self.global_step += 1
    def train_model(self, args):
        ds = TabularDataset(f"dataset/{self.dataset_name}/{self.dataset_name}.csv",
                             int((self.input_dim)**0.5), self.attrib_num, self.label_col)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # start MLflow / W&B
        mlflow.set_experiment("VAE-TableGAN")
        mlflow.start_run()
        wandb.init(project="vae-tablegan", name=self.test_id, config=vars(args))

        self.pre_train(loader)
        self.fine_tune(loader)

        # save final
        self.save()
        mlflow.end_run()
        wandb.finish()