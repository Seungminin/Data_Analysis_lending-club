
import pickle
import pandas as pd
import numpy as np

class DummyTransformer:
    def __init__(self, transformer_path='./transformer.pkl'):
        with open(transformer_path, 'rb') as f:
            self.transformer = pickle.load(f)

    def inverse(self, data):
        return self.transformer.inverse_transform(data)

def inverse_transform(fake_data_np, transformer_path='./transformer.pkl', save_path='./preprocess/generated_restored.csv'):
    transformer = DummyTransformer(transformer_path)
    restored = transformer.inverse(fake_data_np)
    df = pd.DataFrame(restored)
    df.to_csv(save_path, index=False)
    print(f'✅ Restored generated samples saved to {save_path}')
