# main.py
import argparse
import os
import time
import torch
import pandas as pd
import pickle
import wandb

from model.ctabgan import CTABGAN
from model.synthesizer.transformer import DataTransformer
from model.synthesizer.ctabgan_synthesizer import Condvec, Sampler
from model.pipeline.data_preparation import DataPrep


def main():
    parser = argparse.ArgumentParser(description="CTAB-GAN Controller")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--generate", action="store_true")

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ctabgan_epoch_300.pt")
    parser.add_argument("--preprocess_info", type=str, default="checkpoints/preprocess_info_smotified.pkl")
    parser.add_argument("--output_path", type=str, default="Fake_Datasets/smotified_ctabgan.csv")
    parser.add_argument("--input_csv", type=str, default="Real_Datasets/smotifiedCTGAN_data_class1.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    categorical_columns = ['debt_settlement_flag', 'sub_grade', 'home_ownership',
                           'purpose', 'grade', 'term_months']
    log_columns = ['annual_inc', 'revol_bal', 'avg_cur_bal', 'installment']
    mixed_columns = {
        'last_fico_range_high': [0.0],
        'mo_sin_old_rev_tl_op': [0.0],
        'dti': [0.0],
        'revol_util': [0.0]
    }
    integer_columns = ['credit_history_years', 'term_months']
    problem_type = {"Classification": 'loan_status'}

    if args.preprocess:
        raw_df = pd.read_csv(args.input_csv)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_prep = DataPrep(
            raw_df,
            categorical_columns,
            log_columns,
            mixed_columns,
            integer_columns,
            problem_type,
            test_ratio=0.2
        )
        problem_key = list(problem_type.keys())[0]
        target_col = problem_type[problem_key]
        target_index = data_prep.df.columns.get_loc(target_col)

        transformer = DataTransformer(
            train_data=data_prep.df,
            categorical_list=data_prep.column_types['categorical'],
            mixed_dict=mixed_columns
        )
        transformer.fit()
        print("\n[DEBUG] transformer.output_info:")
        for item in transformer.output_info:
            print(item)
        
        transformed = transformer.transform(data_prep.df.values)
        condvec = Condvec(transformed, transformer.output_info,device=device)
        sampler = Sampler(transformed, transformer.output_info, device=device)

        os.makedirs(os.path.dirname(args.preprocess_info), exist_ok=True)

        with open(args.preprocess_info, "wb") as f:
            pickle.dump({
                "data_prep": data_prep,
                "transformer": transformer,
                "condvec": condvec,
                "sampler": sampler,
                "output_info": transformer.output_info,
                "target_index": target_index
            }, f)

        print(f"✅ Preprocessing saved to {args.preprocess_info}")
        return

    if args.train:
        wandb.init(project="CTABGAN", name=f"ctabgan_{int(time.time())}")
        with open(args.preprocess_info, "rb") as f:
            info = pickle.load(f)

        model = CTABGAN(
            raw_csv_path=args.input_csv,
            categorical_columns=categorical_columns,
            log_columns=log_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns,
            problem_type=problem_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        model.fit(
            df=info["data_prep"].df,
            transformer=info["transformer"],
            condvec=info["condvec"],
            sampler=info["sampler"],
            output_info=info["output_info"],
            target_index=info["target_index"]
        )
        print("✅ Training complete.")

    if args.generate:
        with open(args.preprocess_info, "rb") as f:
            info = pickle.load(f)

        model = CTABGAN(
            raw_csv_path=args.input_csv,
            categorical_columns=categorical_columns,
            log_columns=log_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns,
            problem_type=problem_type
        )
        model.synthesizer.transformer = info["transformer"]
        model.synthesizer.cond_generator = info["condvec"]
        model.data_prep = info["data_prep"]
        model.synthesizer.output_info = info["output_info"]

        model.synthesizer._build_generator_only()

        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.synthesizer.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.synthesizer.generator.eval()

        synthetic_df = model.generate_samples(N_CLS_PER_GEN=540000)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        synthetic_df.to_csv(path_or_buf=args.output_path, index=False)
        print(f"✅ Synthetic data saved to {args.output_path}")


if __name__ == "__main__":
    main()
