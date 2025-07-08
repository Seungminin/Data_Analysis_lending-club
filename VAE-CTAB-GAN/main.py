import argparse
import os
import pandas as pd
import torch
import wandb
import pickle
import numpy as np

from model.preprocess import preprocess_data
from model.pipeline.data_utils import load_processed_data, extract_continuous_features
from model.train_loop import train_vae_gan, generate_samples
from model.model import VAEEncoder, Generator, Discriminator, Classifier, VAE_CTABGAN
from model.pipeline.data_utils import show_all_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="VAE-CTAB-GAN")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train", "generate", "only_train"],
                        help="Choose one mode: preprocess, train, generate, only_train")
    parser.add_argument('--z_dim', type=int, default=64, help="z_dimention")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dim for Generator")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for VAE")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL loss weight")
    parser.add_argument("--g_weight", type=float, default=1.0, help="Generator loss weight")
    parser.add_argument("--wandb_project", type=str, default="vae-ctab-gan", help="wandb project name")
    parser.add_argument("--wandb_run", type=str, default="joint-training", help="wandb run name")
    parser.add_argument('--dataset_path', type=str, default='dataset/train_category_1.csv')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='vae_ctabgan.pt')
    parser.add_argument('--preprocessed_path', type=str, default='dataset/preprocessed.csv')
    parser.add_argument('--transformer_path', type=str, default='preprocess/transformer/transformer.pkl')
    parser.add_argument('--num_samples', type=int, default=540000, help='Number of samples to generate')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.preprocessed_path), exist_ok=True)

    if args.mode == "preprocess":
        print("🔄 Preprocessing original data...")
        preprocess_data()
        return

    if not os.path.exists(args.preprocessed_path):
        raise FileNotFoundError("Processed CSV not found. Run with --mode preprocess first.")

    print("📥 Loading processed data...")
    data = load_processed_data(args.preprocessed_path)
    cont_data = extract_continuous_features(data, transformer_path=args.transformer_path)

    if args.mode == "generate":
        print("✨ Generating synthetic samples via generate_samples()...")
        generate_samples(
            args=args,
            full_data=data,
            cont_data=cont_data,
            device=device
        )
        return

    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = CTABGenerator(latent_dim=args.latent_dim).to(device)
    discriminator = CTABDiscriminator(input_dim=data.shape[1]).to(device)
    classifier = CTABClassifier()

    model = VAE_CTABGAN(
        embedding_dim=args.embedding_dim,
        z_dim=args.z_dim,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_dir=args.sample_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    show_all_parameters(model)

    if args.mode in ["train", "only_train"]:
        print(" Starting training...")
        train_vae_gan(
            encoder=encoder,
            generator=generator,
            discriminator=discriminator,
            full_data=data,
            cont_data=cont_data,
            args=args,
            device=device
        )

if __name__ == "__main__":
    main()
