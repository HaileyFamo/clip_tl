# run.py
import argparse
from pathlib import Path

from src.analysis.analyze_lens import main as analyze_main
from src.train.train_clip_tl import main as train_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the script in train or analyze mode.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'analyze', 't', 'a'],
        help="Mode to run the script in. Can be 'train' or 'analyze'.",
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs.yaml',
        help='Path to the YAML configuration file, relative to project root.',
    )
    args, unknown = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent
    if args.mode == 'train' or args.mode == 't':
        train_main(project_root, args.config)
    elif args.mode == 'analyze' or args.mode == 'a':
        analyze_main(project_root, args.config)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
