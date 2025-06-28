import os
import torch
import argparse
import wandb
import mlflow
import pandas as pd
import pickle
from model import VAETableGan
from utils import pp, generate_data, show_all_parameters, CustomDataTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=200)
    parser.add_argument('--lr',          type=float, default=0.0002)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--input_dim',   type=int,   default=6)
    parser.add_argument('--dataset',     type=str,   default='loan_1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--sample_dir',     type=str, default='samples')
    parser.add_argument('--train',       action='store_true')
    parser.add_argument('--test_id',     type=str,   default='test1')
    parser.add_argument('--label_col',   type=int,   default=-1)
    parser.add_argument('--attrib_num',  type=int,   default=15)
    parser.add_argument('--pre_epochs',  type=int,   default=50)
    return parser.parse_args()

def main():
    args = parse_args()

    test_configs = {
        'test1': {
            'lambda_vae':    1.0,
            'lambda_info':   1.0,
            'lambda_advcls': 1.0,
            'delta_mean':    0.0,
            'delta_var':     0.0,
        },
        'test2': {
            'lambda_vae':    1.0,
            'lambda_info':   0.5,
            'lambda_advcls': 0.5,
            'delta_mean':    0.1,
            'delta_var':     0.1,
        },
        'test3': {
            'lambda_vae':    0.3,
            'lambda_info':   0.7,
            'lambda_advcls': 0.7,
            'delta_mean':    0.2,
            'delta_var':     0.2,
        },
    }
    if args.test_id not in test_configs:
        raise ValueError(f"Unknown test_id: {args.test_id}")
    cfg = test_configs[args.test_id]

    args.lambda_vae    = cfg['lambda_vae']
    args.lambda_info   = cfg['lambda_info']
    args.lambda_advcls = cfg['lambda_advcls']
    args.delta_m       = cfg['delta_mean']
    args.delta_v       = cfg['delta_var']

    pp(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAETableGan(
        input_dim     = args.input_dim,
        batch_size    = args.batch_size,
        y_dim         = 2,
        delta_mean    = args.delta_m,
        delta_var     = args.delta_v,
        attrib_num    = args.attrib_num,
        label_col     = args.label_col,
        checkpoint_dir= args.checkpoint_dir,
        sample_dir    = args.sample_dir,
        dataset_name  = args.dataset,
        test_id       = args.test_id,
        device        = device,
        lr            = args.lr,
        pre_epochs    = args.pre_epochs,
        epochs        = args.epoch,
        lambda_vae    = args.lambda_vae,
        lambda_info   = args.lambda_info,
        lambda_advcls = args.lambda_advcls
    )

    show_all_parameters(model)

    if args.train:
        raw_data = pd.read_csv(f"dataset/{args.dataset}/{args.dataset}.csv")
        y = raw_data['loan_status']
        X = raw_data.drop(columns=['loan_status'])

        print(f"[INFO] Raw X shape: {X.shape}, y shape: {y.shape}")

        log_cols = ['annual_inc', 'avg_cur_bal', 'revol_bal', 'installment', 'total_pymnt', 'funded_amnt', 'loan_amnt']

        transformer = CustomDataTransformer(log_features=log_cols)
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        print(f"[INFO] Transformed X shape: {X_transformed.shape}")

        processed = X_transformed.copy()
        processed['loan_status'] = y.values
        save_path = f"dataset/{args.dataset}/{args.dataset}_processed.csv"
        processed.to_csv(save_path, index=False)
        transformer.save(f"{args.sample_dir}/transformer.pkl")

        model.dataset_name = args.dataset

        wandb.init(project="vae-tablegan", name=args.test_id, config=vars(args))
        model.train_model(args)
    else:
        model.load()
        synthetic_data, output_path = generate_data(model, args.sample_dir, num_samples=540000, batch_size=128)

        transformer_path = os.path.join(args.sample_dir, "transformer.pkl")
        if os.path.exists(transformer_path):
            print("[INFO] Applying inverse_transform using saved transformer...")
            with open(transformer_path, 'rb') as f:
                transformer = pickle.load(f)
            try:
                synthetic_data = transformer.inverse_transform(synthetic_data)
                print("[INFO] inverse_transform applied successfully.")
            except Exception as e:
                print(f"[WARN] inverse_transform failed: {e}")
        else:
            print("[WARN] No transformer.pkl found. Saving raw transformed data.")

        # 💾 Save
        synthetic_data.to_csv(output_path, index=False)
        print(f"[INFO] Generated data saved to: {output_path}")

if __name__ == '__main__':
    main()
