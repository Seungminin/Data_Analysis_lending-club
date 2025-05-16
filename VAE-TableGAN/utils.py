# utils.py
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def pp(obj):
    from pprint import pprint
    pprint(obj)

def show_all_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

def generate_data(model, save_dir, num_samples=10000):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 🔹 1. Load original dataset to match structure
    data_path = f'dataset/{model.dataset_name}/{model.dataset_name}.csv'
    original = pd.read_csv(data_path)
    original = original.drop(columns=['loan_status'])  
    feature_names = original.columns.tolist()

    # 🔹 2. Create latent noise and generate synthetic data
    z = torch.randn(num_samples, model.latent_dim).to(model.device)
    with torch.no_grad():
        fake = model.generator(z).cpu().numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(original)
    fake_inverse = scaler.inverse_transform(fake)

    for i, col in enumerate(feature_names):
        real_unique = np.sort(original[col].unique())
        indices = np.searchsorted(real_unique, fake_inverse[:, i], side="left")
        indices = np.clip(indices, 0, len(real_unique) - 1)
        fake_inverse[:, i] = real_unique[indices]

    output_path = os.path.join(save_dir, f"{model.test_id}_generated.csv")
    pd.DataFrame(fake_inverse, columns=feature_names).to_csv(output_path, index=False)

    print(f"[+] Generated data saved to {output_path}")

def padding_duplicating(data, row_size):
    arr_data = np.array(data.values.tolist())
    col_num = arr_data.shape[1]
    npad = ((0, 0), (0, row_size - col_num))
    arr_data = np.pad(arr_data, pad_width=npad, mode='constant', constant_values=0.)

    for i in range(1, arr_data.shape[1] // col_num):
        arr_data[:, col_num * i: col_num * (i + 1)] = arr_data[:, 0: col_num]

    return arr_data

def reshape(data, dim=None):
    return data.values.reshape(data.shape[0], -1)


