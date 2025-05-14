# utils.py
import torch
import os
import pandas as pd
import numpy as np

def pp(obj):
    from pprint import pprint
    pprint(obj)

def show_all_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

def generate_data(model, save_dir, num_samples=10000):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    z = torch.randn(num_samples, model.latent_dim).to(model.device)
    with torch.no_grad():
        fake = model.generator(z).cpu().numpy()
    path = os.path.join(save_dir, f"{model.test_id}_generated.csv")
    pd.DataFrame(fake).to_csv(path, index=False)
    print(f"[+] Generated data saved to {path}")
