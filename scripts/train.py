"""Train the final Med-UniCAD MoFX-Dense model."""

from __future__ import annotations

import argparse

from medunicad.configuration import load_training_config
from medunicad.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final MoFX-Dense model with paper settings."
    )
    parser.add_argument(
        "--config",
        default="configs/paper_training.json",
        help="Path to the paper training configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        choices=(42, 123, 2026),
        help="Paper seed to reproduce.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_training_config(args.config), seed=args.seed)


if __name__ == "__main__":
    main()
