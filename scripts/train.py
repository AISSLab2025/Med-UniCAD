"""Train the final Med-UniCAD MoFX-Dense model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Permit direct execution from a source checkout before an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medunicad.configuration import load_training_config


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
    parser.add_argument(
        "--resume-run",
        metavar="RUN_ID",
        help=(
            "Resume an existing outputs/runs/<RUN_ID> directory, for example "
            "2026-10-27_00-39-54."
        ),
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Skip training and rerun final validation calibration plus testing "
            "from the existing run's best_model.pth. Requires --resume-run."
        ),
    )
    args = parser.parse_args()
    if args.test_only and not args.resume_run:
        parser.error("--test-only requires --resume-run RUN_ID")
    return args


def main() -> None:
    args = parse_args()
    # Delay heavy ML imports until after argument parsing so --help remains
    # available even when optional training dependencies are not installed.
    from medunicad.training import train

    train(
        load_training_config(args.config),
        seed=args.seed,
        resume_run_id=args.resume_run,
        test_only=args.test_only,
    )


if __name__ == "__main__":
    main()
