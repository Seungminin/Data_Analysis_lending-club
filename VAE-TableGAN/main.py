import os
import torch
import argparse
import wandb
import mlflow
from model import VAETableGan
from utils import pp, generate_data, show_all_parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=64)  # will be treated as image width/height
    parser.add_argument('--dataset', type=str, default='loan_1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test_id', type=str, default='test1')
    parser.add_argument('--label_col', type=int, default=-1)
    parser.add_argument('--attrib_num', type=int, default=15)
    return parser.parse_args()

def main():
    args = parse_args()

    # 실험별 하이퍼파라미터 설정
    test_configs = {
        'test1': {'alpha': 1.0, 'beta': 1.0, 'delta_m': 0.0, 'delta_v': 0.0},
        'test2': {'alpha': 0.5, 'beta': 0.5, 'delta_m': 0.1, 'delta_v': 0.1},
        'test3': {'alpha': 0.3, 'beta': 0.7, 'delta_m': 0.2, 'delta_v': 0.2}
    }
    if args.test_id not in test_configs:
        raise ValueError("Invalid test_id provided")
    cfg = test_configs[args.test_id]
    args.alpha, args.beta = cfg['alpha'], cfg['beta']
    args.delta_m, args.delta_v = cfg['delta_m'], cfg['delta_v']

    pp(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAETableGan(
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        y_dim=2,
        alpha=args.alpha,
        beta=args.beta,
        delta_mean=args.delta_m,
        delta_var=args.delta_v,
        attrib_num=args.attrib_num,
        label_col=args.label_col,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        dataset_name=args.dataset,
        test_id=args.test_id,
        device=device,
        lr=args.lr,
        epochs=args.epoch
    )

    if args.train:
        wandb.init(project="vae-tablegan", name=args.test_id, config=vars(args))
        model.train_model()
    else:
        model.load()
        generate_data(model, args.sample_dir, num_samples=540000, batch_size=128)

if __name__ == '__main__':
    main()
