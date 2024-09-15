import argparse
import os
from core.solver import Solver
from core.data_loader import get_train_loader, get_test_loader, get_eval_loader
import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ('true', '1')


def main(args):
    # Data loaders
    if args.mode == 'train':
        loaders = Munch(
            src=get_train_loader(root=args.train_img_dir, which='source',
                                 img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers),
            ref=get_train_loader(root=args.train_img_dir, which='reference',
                                 img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers),
            val=get_eval_loader(root=args.val_img_dir,
                                img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers))
    elif args.mode == 'sample':
        loaders = Munch(
            src=get_test_loader(root=args.src_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers),
            ref=get_test_loader(root=args.ref_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers))
    else:
        loaders = None

    solver = Solver(args)

    if args.mode == 'train':
        solver.train(loaders)
    elif args.mode == 'sample':
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--style_dim', type=int, default=64, help='Dimension of style code')
    parser.add_argument('--latent_dim', type=int, default=16, help='Dimension of latent code')
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--w_hpf', type=float, default=1, help='Weight for high-pass filtering')

    # Training configuration
    parser.add_argument('--total_iters', type=int, default=200000, help='Total iterations for training')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training from')
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for the mapping network')
    parser.add_argument('--beta1', type=float, default=0.0, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1, help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_percep', type=float, default=0.1, help='Weight for perceptual loss')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Weight for gradient penalty in WGAN-GP')

    # Directories
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val', help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory for saving samples')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory for saving test results')
    parser.add_argument('--src_dir', type=str, default='data/celeba_hq/test/src', help='Directory containing source test images')
    parser.add_argument('--ref_dir', type=str, default='data/celeba_hq/test/ref', help='Directory containing reference test images')

    # Step sizes
    parser.add_argument('--print_every', type=int, default=1000, help='Interval for printing training logs')
    parser.add_argument('--sample_every', type=int, default=1000, help='Interval for saving image samples')
    parser.add_argument('--save_every', type=int, default=1000, help='Interval for saving model checkpoints')
    parser.add_argument('--eval_every', type=int, default=5000, help='Interval for evaluation')

    # Miscellaneous
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='Mode of operation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    main(args)
