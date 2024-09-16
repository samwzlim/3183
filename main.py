import argparse
import os
from munch import Munch
from core.solver import Solver
from data_loader import get_train_loader, get_eval_loader, get_test_loader


def parse_args():
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val', help='Directory containing validation images')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src', help='Source directory for test images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref', help='Reference directory for test images')

    # Training parameters
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--total_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--resume_iter', type=int, default=0, help='Resume training from this iteration')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for mapping network')
    parser.add_argument('--beta1', type=float, default=0.0, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')

    # Loss hyperparameters
    parser.add_argument('--lambda_sty', type=float, default=1.0, help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_cyc', type=float, default=10.0, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_ds', type=float, default=1.0, help='Weight for diversity-sensitive loss')

    # New hyperparameters for perceptual and identity loss
    parser.add_argument('--lambda_perc', type=float, default=10.0, help='Weight for perceptual loss')
    parser.add_argument('--lambda_id', type=float, default=5.0, help='Weight for identity loss')

    # Misc
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'eval'], help='Mode of operation: train, sample, eval')
    parser.add_argument('--sample_dir', type=str, default='expr/samples', help='Directory for saving generated samples')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints', help='Directory for saving model checkpoints')
    parser.add_argument('--result_dir', type=str, default='expr/results', help='Directory for saving results')
    parser.add_argument('--eval_dir', type=str, default='expr/eval', help='Directory for saving evaluation metrics')
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.pth', help='Path to pre-trained FAN model')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz', help='Path to precomputed CelebA landmark mean')
    parser.add_argument('--randcrop_prob', type=float, default=0.5, help='Probability of applying random crop')
    parser.add_argument('--w_hpf', type=int, default=1, help='Weight for high-pass filtering')
    parser.add_argument('--print_every', type=int, default=1000, help='Print logs every N iterations')
    parser.add_argument('--sample_every', type=int, default=5000, help='Save sample images every N iterations')
    parser.add_argument('--save_every', type=int, default=10000, help='Save model checkpoints every N iterations')
    parser.add_argument('--eval_every', type=int, default=50000, help='Evaluate FID and LPIPS every N iterations')

    return parser.parse_args()


def main(args):
    # Initialize the solver
    solver = Solver(args)

    # Choose the mode of operation
    if args.mode == 'train':
        # Training mode
        loaders = Munch(src=get_train_loader(args.train_img_dir, which='source', img_size=args.img_size,
                                             batch_size=args.batch_size, prob=args.randcrop_prob, num_workers=args.num_workers),
                        ref=get_train_loader(args.train_img_dir, which='reference', img_size=args.img_size,
                                             batch_size=args.batch_size, prob=args.randcrop_prob, num_workers=args.num_workers),
                        val=get_eval_loader(args.val_img_dir, img_size=args.img_size, batch_size=args.val_batch_size))

        # Start training
        solver.train(loaders)

    elif args.mode == 'sample':
        # Sampling mode
        loaders = Munch(src=get_test_loader(args.src_dir, img_size=args.img_size, batch_size=args.val_batch_size),
                        ref=get_test_loader(args.ref_dir, img_size=args.img_size, batch_size=args.val_batch_size))
        
        solver.sample(loaders)

    elif args.mode == 'eval':
        # Evaluation mode
        solver.evaluate()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    main(args)
