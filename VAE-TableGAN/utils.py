# utils.py
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import joblib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import wasserstein_distance

class CustomDataTransformer:
    def __init__(self, log_features=None):
        self.log_features = log_features or []
        self.label_encoders = {}
        self.scaler = None
        self.columns = None

    def fit(self, df):
        df = df.copy()
        self.columns = df.columns.tolist()

        # 1. Log transform
        for col in self.log_features:
            df[col] = np.log1p(df[col])

        # 2. Label encoding
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # 3. MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(df)
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.log_features:
            df[col] = np.log1p(df[col])
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return pd.DataFrame(self.scaler.transform(df), columns=self.columns)

    def inverse_transform(self, arr):
        df = pd.DataFrame(self.scaler.inverse_transform(arr), columns=self.columns)
        for col, le in self.label_encoders.items():
            df[col] = np.round(df[col]).astype(int)
            df[col] = le.inverse_transform(df[col])
        for col in self.log_features:
            df[col] = np.expm1(df[col])
        return df

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

class TabularDataset(Dataset):
    def __init__(self, csv_path, input_dim, attrib_num, label_col=-1):
        df = pd.read_csv(csv_path)
        self.y = df['loan_status'].values.astype(int)
        df = df.drop(columns=['loan_status'])

        self.X_padded = padding_duplicating(df, input_dim * input_dim)
        self.input_dim = input_dim

    def __len__(self):
        return len(self.X_padded)

    def __getitem__(self, idx):
        x = self.X_padded[idx].reshape(self.input_dim, self.input_dim).astype(np.float32)
        return torch.tensor(x).unsqueeze(0), torch.tensor(self.y[idx]).long()
    
def pp(obj):
    from pprint import pprint
    pprint(obj)

def show_all_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")


def compute_mmd(x, y, gamma=1.0):
    """RBF kernel MMD between x,y torch tensors of shape [n, d]."""
    xx = torch.exp(-gamma * ((x[:,None]-x[None,:])**2).sum(-1))
    yy = torch.exp(-gamma * ((y[:,None]-y[None,:])**2).sum(-1))
    xy = torch.exp(-gamma * ((x[:,None]-y[None,:])**2).sum(-1))
    return xx.mean() + yy.mean() - 2*xy.mean()

def compute_wasserstein(x, y):
    """평균 feature-wise Wasserstein distance between numpy arrays x,y shape [n, d]."""
    return np.mean([wasserstein_distance(x[:,i], y[:,i]) for i in range(x.shape[1])])


def generate_data(model, save_dir, num_samples=10000, batch_size = 64):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 🔹 1. Load original dataset to match structure
    data_path = f'dataset/{model.dataset_name}/{model.dataset_name}.csv'
    original = pd.read_csv(data_path)
    original = original.drop(columns=['loan_status'])  
    feature_names = original.columns.tolist()

    transformer = CustomDataTransformer.load(f"{save_dir}/transformer.pkl")
    all_generated = []

    for i in tqdm(range(0, num_samples, batch_size), desc = 'Generate Samples'):
        current_batch = min(batch_size, num_samples-1)

        # 🔹 2. Create latent noise and generate synthetic data
        z = torch.randn(current_batch, model.latent_dim).to(model.device)
        with torch.no_grad():
            fake = model.generator(z).cpu().numpy()

        # 🔹 3. Postprocess: reshape and inverse scale
        fake = fake.squeeze()  # shape: (N, H, W)
        if len(fake.shape) == 3:
            fake = fake.reshape(fake.shape[0], -1)  # (N, H×W)

        fake_part_df = transformer.inverse_transform(fake[:, :len(feature_names)])
        fake_part = fake_part_df.values  # NumPy array로 변환

        for j, col in enumerate(feature_names):
            real_unique = np.sort(original[col].unique())
            indices = np.searchsorted(real_unique, fake_part[:, j], side="left")
            indices = np.clip(indices, 0, len(real_unique) - 1)
            fake_part[:, j] = real_unique[indices]
        all_generated.append(fake_part)
    
    full_generated = np.vstack(all_generated)
    output_path = os.path.join(save_dir, f"{model.dataset_name}_{model.test_id}_{model.pre_epochs}_generated_CNNEncoder.csv")
    synthetic_data = pd.DataFrame(full_generated, columns=feature_names)

    print(f"[+] return data_path {output_path}")

    return synthetic_data, output_path

def padding_duplicating(data, row_size):
    arr_data = np.array(data.values.tolist())
    col_num = arr_data.shape[1]
    npad = ((0, 0), (0, row_size - col_num))
    arr_data = np.pad(arr_data, pad_width=npad, mode='constant', constant_values=0.)

    for i in range(1, arr_data.shape[1] // col_num):
        arr_data[:, col_num * i: col_num * (i + 1)] = arr_data[:, 0: col_num]

    return arr_data

def reshape(data, dim=None):
    if dim:
        return data.values.reshape(data.shape[0], dim, dim)
    else:
        return data.values.reshape(data.shape[0], -1)
    

def visualize_latent_vectors(encoder, dataloader, device, method='tsne'):
    encoder.eval()
    all_z, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z, _, _ = encoder(x)
            all_z.append(z.cpu())
            all_labels.append(y.cpu())

    all_z = torch.cat(all_z).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 2D Reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method should be 'tsne' or 'pca'")

    z_2d = reducer.fit_transform(all_z)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
