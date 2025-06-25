import os
import torch
import argparse
import wandb
import pandas as pd
import joblib
from model import VAE_CTGAN
from data_transformer import DataTransformer
from utils import show_all_parameters, load_transformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_e', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dataset_path', type=str, default='dataset/train_category_1.csv')
    parser.add_argument('--discriminator_steps', type=int, default=1)
    parser.add_argument('--pac', type=int, default=10)
    parser.add_argument('--log_frequency', action='store_true')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='vae_ctgan.pt')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--preprocessed_path', type=str, default='dataset/preprocessed.csv')
    parser.add_argument('--transformer_path', type=str, default='dataset/transformer.pkl')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.preprocessed_path), exist_ok=True)

    model = VAE_CTGAN(
        embedding_dim=args.embedding_dim,
        z_dim=args.z_dim,
        device=device,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lr_e=args.lr_e,
        discriminator_steps=args.discriminator_steps,
        pac=args.pac,
        sample_dir=args.sample_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_frequency=args.log_frequency
    )

    show_all_parameters(model)

    if args.preprocess:
        df = pd.read_csv(args.dataset_path)
        discrete_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        transformer = DataTransformer()
        transformer.fit(df, discrete_cols)
        transformed_data = transformer.transform(df)

        # Save
        pd.DataFrame(transformed_data).to_csv(args.preprocessed_path, index=False)
        joblib.dump(transformer, args.transformer_path)
        print(f"✅ Preprocessing complete. Saved to {args.preprocessed_path} and {args.transformer_path}")
        return

    elif args.train:
        wandb.init(project="vae-ctgan", config=vars(args))

        # Load preprocessed data and transformer
        transformed_df = pd.read_csv(args.preprocessed_path)
        model.train_from_transformed(
            data_path=args.preprocessed_path,
            transformer_path=args.transformer_path,
            epochs=args.epochs
        )

    else:
        model.load(os.path.join(args.checkpoint_dir, args.save_name))
        model._transformer = load_transformer(args.transformer_path)
        df = model.sample(n=540000)
        print(df.head())

if __name__ == "__main__":
    main()



"""
--preprocess : 데이터 전처리, categorical은 one-hot encoding, continuous GMM 모드 정규화
-- train + train_from_transformed : 전처리된 데이터를 학습한다.
-- train + fit_from_dataframe : 데이터 전처리부터 학습 진행
"""