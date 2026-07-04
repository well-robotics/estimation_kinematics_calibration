"""Command-line entry point for B1 bi-level calibration."""

from __future__ import annotations

import importlib

from .config import BilevelConfig


def _require_runtime_dependencies() -> None:
    missing = []
    for module in ("casadi", "pinocchio", "cvxpy"):
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install the conda/pip packages described in README.md."
        )

    import casadi as cs

    if not cs.has_nlpsol("fatrop"):
        raise RuntimeError(
            "CasADi is installed, but the Fatrop NLP plugin is not available. "
            "Install a CasADi/Fatrop build that provides nlpsol('fatrop')."
        )


def main() -> None:
    _require_runtime_dependencies()

    from .calibration import FrankWolfeCalibrator
    from .codegen import CodegenLibraryLoader
    from .data_io import CsvDatasetLoader
    from .robot import B1RobotModel

    config = BilevelConfig()
    dataset = CsvDatasetLoader(config).load()
    robot = B1RobotModel.from_config(config)
    codegen = CodegenLibraryLoader(config.external_lib_dir).load()
    FrankWolfeCalibrator(config, dataset, robot, codegen).run()


if __name__ == "__main__":
    main()
