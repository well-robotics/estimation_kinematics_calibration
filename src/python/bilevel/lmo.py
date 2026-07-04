"""Linear minimization oracle for the Frank-Wolfe upper level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FrankWolfeConfig, WeightParameterLayout


@dataclass(frozen=True)
class LMOResult:
    point: np.ndarray
    status: str
    objective: float


class LinearMinimizationOracle:
    """CVXPY formulation of the feasible-set LMO."""

    def __init__(self, layout: WeightParameterLayout, config: FrankWolfeConfig):
        self.layout = layout
        self.config = config

    def solve(self, gradient: np.ndarray, theta: np.ndarray) -> LMOResult:
        import cvxpy as cp

        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if gradient.size != self.layout.total_size:
            raise ValueError(
                f"gradient has size {gradient.size}; expected {self.layout.total_size}"
            )
        if theta.size != self.layout.total_size:
            raise ValueError(
                f"theta has size {theta.size}; expected {self.layout.total_size}"
            )

        x_opt = cp.Variable(self.layout.total_size)
        constraints = self._constraints(cp, x_opt, theta)
        problem = cp.Problem(cp.Minimize(gradient @ x_opt), constraints)

        solver = getattr(cp, self.config.lmo_solver, self.config.lmo_solver)
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:
            # CLARABEL is a useful open-source fallback for development machines.
            if self.config.lmo_solver == "CLARABEL":
                raise
            problem.solve(solver=getattr(cp, "CLARABEL", "CLARABEL"), verbose=False)

        if x_opt.value is None:
            raise RuntimeError(f"LMO failed with status {problem.status}")
        return LMOResult(
            point=np.asarray(x_opt.value, dtype=float).reshape(-1),
            status=str(problem.status),
            objective=float(problem.value),
        )

    def variable_bounds(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return scalar lower/upper bounds without PSD constraints."""

        cfg = self.config
        layout = self.layout
        theta = np.asarray(theta, dtype=float).reshape(-1)

        lb = -cfg.big_box * np.ones(layout.total_size)
        ub = cfg.big_box * np.ones(layout.total_size)

        lb[layout.arrival_slice] = cfg.core_min
        ub[layout.arrival_slice] = cfg.core_max

        for block in (
            *layout.measurement_covariance_slices,
            *layout.noise_covariance_slices,
        ):
            lb[block.start : block.start + 3] = cfg.eps_diag
            ub[block.start : block.start + 3] = cfg.core_max
            lb[block.start + 3 : block.stop] = -cfg.core_max
            ub[block.start + 3 : block.stop] = cfg.core_max

        lb[layout.random_walk_slice] = cfg.eps_diag
        ub[layout.random_walk_slice] = cfg.core_max

        lb[layout.swing_slice] = cfg.qswing_min
        ub[layout.swing_slice] = max(cfg.qswing_max, float(np.max(theta[layout.swing_slice])))

        lb[layout.stance_slice] = cfg.qstance_min
        ub[layout.stance_slice] = max(cfg.qstance_max, float(np.max(theta[layout.stance_slice])))

        lb[layout.tip_slice] = -cfg.tip_bound
        ub[layout.tip_slice] = cfg.tip_bound
        lb[layout.base_slice] = -cfg.base_bound
        ub[layout.base_slice] = cfg.base_bound
        return lb, ub

    def _constraints(self, cp, x_opt, theta: np.ndarray) -> list:
        cfg = self.config
        layout = self.layout
        idx_noise = layout.noise_slice.start

        lb, ub = self.variable_bounds(theta)

        adaptive_box = cfg.adaptive_abs_box_scale * np.maximum(1.0, np.abs(theta))
        adaptive_box = np.maximum(adaptive_box, np.abs(ub))
        distance_to_box = np.maximum(0.0, np.maximum(lb - theta, theta - ub))
        trust_region = np.maximum(cfg.trust_region_radius, distance_to_box)

        constraints = [
            cp.abs(x_opt) <= adaptive_box,
            cp.abs(x_opt - theta) <= trust_region,
            x_opt >= lb,
            x_opt <= ub,
        ]

        meas = layout.measurement_slice.start
        for offset in (meas, meas + 6, idx_noise, idx_noise + 6):
            block = self._symmetric_3x3(cp, x_opt[offset : offset + 6])
            constraints += [
                block >> cfg.eps_psd * np.eye(3),
                cp.trace(block) <= cfg.trace_cap,
            ]

        constraints += [
            x_opt[idx_noise + 12 : idx_noise + 15] >= cfg.eps_diag,
            x_opt[idx_noise + 15 : idx_noise + 18] >= cfg.eps_diag,
            x_opt[idx_noise + 18 : idx_noise + 21] >= cfg.eps_diag,
            x_opt[idx_noise + 21 : idx_noise + 24] >= cfg.eps_diag,
        ]
        return constraints

    @staticmethod
    def _symmetric_3x3(cp, v6):
        return cp.bmat(
            [
                [v6[0], v6[3], v6[4]],
                [v6[3], v6[1], v6[5]],
                [v6[4], v6[5], v6[2]],
            ]
        )
