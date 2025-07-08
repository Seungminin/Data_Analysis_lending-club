import argparse
import os
import pandas as pd
import torch
import wandb

from preprocess import preprocess_data
from data_utils import load_processed_data, extract_continuous_features
from train_loop import train_vae_gan
from model import VAEEncoder, Generator, Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="VAE-CTAB-GAN")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train", "generate", "only_train"],
                        help="Choose one mode: preprocess, train, generate, only_train")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for VAE")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL loss weight")
    parser.add_argument("--g_weight", type=float, default=1.0, help="Generator loss weight")
    parser.add_argument("--wandb_project", type=str, default="vae-ctab-gan", help="wandb project name")
    parser.add_argument("--wandb_run", type=str, default="joint-training", help="wandb run name")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "preprocess":
        print("🔄 Preprocessing original data...")
        preprocess_data()
        return

    # Load processed data
    processed_path = "./preprocess/processed.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError("🛑 Processed CSV not found. Run with --mode preprocess first.")

    data = load_processed_data(processed_path)
    cont_data = extract_continuous_features(data)

    if args.mode == "generate":
        print("🚧 [TODO] Generation logic not implemented yet.")
        return

    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # Models
    encoder = VAEEncoder(input_dim=cont_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator(input_dim=data.shape[1]).to(device)

    if args.mode in ["train", "only_train"]:
        print("🚀 Starting training...")
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