import argparse
import os
import torch
from model.ctabgan import CTABGAN

def main():
    parser = argparse.ArgumentParser(description="CTAB-GAN Controller")
    parser.add_argument("--train", action="store_true", help="Train the CTAB-GAN model")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic samples using trained model")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ctabgan_epoch_300.pt", help="Path to saved .pt model")
    parser.add_argument("--output_path", type=str, default="Fake_Datasets/smotified_ctabgan.csv", help="Path to save generated samples")
    parser.add_argument("--input_csv", type=str, default="Real_Datasets/smotifiedCTGAN_data_class1.csv", help="Path to input CSV for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")

    args = parser.parse_args()

    if args.train:
        model = CTABGAN(
            raw_csv_path=args.input_csv,
            epochs=args.epochs
        )
        model.fit()

    if args.generate:
        model = CTABGAN(
            raw_csv_path=args.input_csv,
            epochs=0  
        )

        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.generator.eval()

        synthetic_df = model.generate_samples(N_CLS_PER_GEN=540000)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        synthetic_df.to_csv(path_or_buf=args.output_path, index=False)
        print(f" Synthetic data saved to {args.output_path}")

if __name__ == "__main__":
    main()
