"""Command-line interface: solve, calibrate, and select."""

from __future__ import annotations

import argparse
import json


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="g1cal",
        description=(
            "Bilevel covariance calibration for a G1 humanoid state "
            "estimator (lower level built on the PRIME estimator)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser(
        "solve", help="lower-level solve for one clip at a covariance"
    )
    solve.add_argument("--clip", choices=("run1", "run2"), required=True)
    solve.add_argument(
        "--covariance", default="calibrated",
        help=("'calibrated', 'baseline', a precision.csv path, or a theta13 "
              "float value"),
    )
    solve.add_argument("--out", default="out/calibration")

    calibrate = sub.add_parser(
        "calibrate", help="run one upper calibration method"
    )
    calibrate.add_argument(
        "--optimizer", choices=("sqp-bfgs", "frank-wolfe-sdp"), required=True
    )
    calibrate.add_argument("--max-iterations", type=int, default=2)
    calibrate.add_argument("--out", default="out/calibration")

    select = sub.add_parser(
        "select", help="select the lowest strict evaluated covariance"
    )
    select.add_argument("--out", default="out/calibration")

    args = parser.parse_args()

    if args.command == "solve":
        return _solve(args)
    if args.command == "calibrate":
        from .optimizers import run_optimizer

        _print(
            run_optimizer(
                algorithm=args.optimizer.replace("-", "_"),
                max_iterations=args.max_iterations,
                output_root=args.out,
            )
        )
        return 0
    if args.command == "select":
        from .optimizers import select_best_feasible

        _print(select_best_feasible(output_root=args.out))
        return 0
    raise AssertionError("unreachable")


def _resolve_theta(covariance: str):
    import numpy as np

    from .calibration import (
        RELEASED_INDEX,
        calibrated_theta,
        initial_theta,
    )
    from .covariance import CovarianceParameterization
    from .paths import resolve_inside_root

    if covariance == "calibrated":
        return calibrated_theta()
    if covariance == "baseline":
        return initial_theta()
    candidate = resolve_inside_root(covariance, must_exist=False)
    if candidate.is_file():
        lines = candidate.read_text().splitlines()
        if len(lines) != 4 or not lines[0].startswith("# config_hash="):
            raise ValueError("precision file must use the g1cal four-line format")
        rows = {}
        for line in lines[1:]:
            label, *values = line.split(",")
            rows[label] = np.asarray([float(value) for value in values])
        expected_widths = {"p0": 70, "q": 35, "r": 70}
        if set(rows) != set(expected_widths) or any(
            rows[name].shape != (width,)
            for name, width in expected_widths.items()
        ):
            raise ValueError("precision file has invalid matrix rows")
        expected_hash = lines[0].split("=", 1)[1]
        parameterization = CovarianceParameterization()
        for theta in (calibrated_theta(), initial_theta()):
            evaluated = parameterization.evaluate(theta)
            if evaluated.config_hash != expected_hash:
                continue
            expected_rows = {
                "p0": evaluated.precision_diag["P0"],
                "q": evaluated.precision_diag["Q"],
                "r": evaluated.precision_diag["R"],
            }
            if not all(np.allclose(rows[name], expected_rows[name], rtol=1e-14,
                                   atol=0.0) for name in rows):
                raise ValueError("precision file values do not match its hash")
            return theta.copy()
        raise ValueError(
            "precision hash is not the released calibrated or baseline value; "
            "pass theta[13] explicitly for another covariance"
        )
    theta = initial_theta()
    theta[RELEASED_INDEX] = float(covariance)
    return theta


def _solve(args) -> int:
    from .calibration import CalibrationOracle

    oracle = CalibrationOracle(output_root=args.out)
    theta = _resolve_theta(args.covariance)
    component = oracle.evaluate_component(
        theta, args.clip, label=f"cli_solve_{args.clip}"
    )
    _print(
        {
            "clip": args.clip,
            "theta13": float(theta[13]),
            "se3_log_loss": component.loss.value,
            "selected_attempt": component.selected_attempt,
            "cache_hit": component.cache_hit,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
