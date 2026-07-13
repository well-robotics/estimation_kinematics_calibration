"""Command-line interface for estimation calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import calibrate, evaluate, inspect_run
from .covariance_calibration import CalibrationConfig
from .data import load_dataset


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="estimation-calibration-cuda")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="calibrate on train and validation splits")
    train.add_argument("data", help="dataset directory or the literal example")
    train.add_argument("-o", "--output", required=True, help="run directory")
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=_positive_int, default=20)
    train.add_argument("--chunk", type=_positive_int, default=300)
    train.add_argument("--lr", type=_positive_float, default=1e-2)
    train.add_argument("--compile", dest="compile_mode", default="auto",
                       choices=["auto", "none", "default", "cuda-graph",
                                "cuda-graph-compile"])
    train.add_argument("--resume", action="store_true")

    test = commands.add_parser("evaluate", help="evaluate the frozen test split once")
    test.add_argument("data", help="dataset directory or the literal example")
    test.add_argument("--checkpoint", required=True, type=Path)
    test.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")

    inspect = commands.add_parser("inspect", help="validate a run and its hashes")
    inspect.add_argument("run", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = calibrate(
                load_dataset(args.data),
                CalibrationConfig(
                    device=args.device,
                    seed=args.seed,
                    epochs=args.epochs,
                    chunk=args.chunk,
                    lr=args.lr,
                    compile_mode=args.compile_mode,
                ),
                output_dir=args.output,
                resume=args.resume,
            )
            output = {
                "run": str(result.run_dir),
                "selected_epoch": result.selected_epoch,
                "validation_body_velocity_rmse_mps": (
                    result.selected_validation_body_velocity_rmse_mps),
            }
        elif args.command == "evaluate":
            output = dict(evaluate(
                load_dataset(args.data), checkpoint=args.checkpoint,
                split="test", device=args.device))
        else:
            output = inspect_run(args.run)
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
