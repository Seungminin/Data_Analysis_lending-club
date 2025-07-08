
import os
import pandas as pd
import numpy as np

def load_processed_data(path='./preprocess/processed.csv'):
    print(f"📄 Loading processed data from {path}...")
    return pd.read_csv(path).astype(np.float32).values


def extract_continuous_features(full_data, transformer_path='./transformer.pkl'):
    # For now, assume continuous features are always at fixed positions based on 'tanh' in output_info
    # Later: can improve by loading output_info via pickle from transformer if saved
    raise NotImplementedError("You must provide the continuous feature indices manually or from transformer.")
