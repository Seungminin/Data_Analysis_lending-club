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