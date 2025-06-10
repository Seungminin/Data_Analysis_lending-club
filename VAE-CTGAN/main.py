import os
import torch
import argparse
import wandb
from model import VAE_CTGAN
from utils import show_all_parameters


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
    parser.add_argument('--dataset_path', type=str, default='train_category_1.csv')
    parser.add_argument('--discriminator_steps', type=int, default=1)
    parser.add_argument('--pac', type=int, default=10)
    parser.add_argument('--log_frequency', action='store_true')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='vae_ctgan.pt')
    parser.add_argument('--train', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project="vae-ctgan", config=vars(args))
    device = torch.device(args.device)

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

    if args.train:
        model.fit(args.dataset_path, epochs=args.epochs)
    else:
        model.load(os.path.join(args.checkpoint_dir, args.save_name))
        samples = model.sample(n=540000)
        print(samples.head())


if __name__ == "__main__":
    main()
