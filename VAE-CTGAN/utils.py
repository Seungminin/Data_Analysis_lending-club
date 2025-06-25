import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TabularDataset(Dataset):
    def __init__(self, path, input_dim, attrib_num, label_col=-1):
        df = pd.read_csv(path)
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = scaler.fit_transform(df)
        self.X = self.pad(self.X, input_dim)

    def pad(self, data, dim):
        padded = torch.zeros((data.shape[0], dim))
        padded[:, :data.shape[1]] = torch.tensor(data, dtype=torch.float32)
        return padded

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def show_all_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

def load_transformer(path):
    import joblib
    return joblib.load(path)
