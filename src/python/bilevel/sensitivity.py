"""KKT-based estimator sensitivities for the upper-level gradient."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from casadi import DM

from .codegen import CodegenFunctions
from .robot import B1RobotModel, MeasurementBundle


@dataclass(frozen=True)
class SensitivityResult:
    dstate_dtheta: np.ndarray
    kkt_inf_norm: float


def _casadi_to_csc(matrix, shape: tuple[int, int]) -> sp.csc_matrix:
    sparsity = matrix.sparsity()
    rows = np.asarray(sparsity.row(), dtype=np.int32)
    colptr = np.asarray(sparsity.colind(), dtype=np.int32)
    values = np.asarray(matrix.nonzeros(), dtype=np.float64)
    return sp.csc_matrix((values, rows, colptr), shape=shape)


def _solve_sparse(matrix: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        from pypardiso import spsolve

        try:
            return spsolve(matrix, rhs, matrix_type=11)
        except TypeError:
            return spsolve(matrix, rhs)
    except Exception:
        from scipy.sparse.linalg import spsolve

        return spsolve(matrix, rhs)


class EstimatorSensitivity:
    """Differentiates the FIE solution through the KKT system."""

    def __init__(self, estimator, robot: B1RobotModel, codegen: CodegenFunctions):
        self.estimator = estimator
        self.robot = robot
        self.codegen = codegen

    def solve(
        self,
        state_traj: np.ndarray,
        noise_traj: np.ndarray,
        costate_traj: np.ndarray,
        measurement: MeasurementBundle,
        controls: np.ndarray,
        contacts: np.ndarray,
        prior: np.ndarray,
        theta_core: np.ndarray,
        g_meas: np.ndarray,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        u_meas: np.ndarray,
    ) -> SensitivityResult:
        horizon = g_meas.shape[0] - 1
        n_state = self.estimator.n_state
        x_vec = np.asarray(state_traj, dtype=float).reshape(-1, 1)
        w_vec = np.asarray(noise_traj, dtype=float).reshape(-1, 1)
        lambda_vec = np.asarray(costate_traj, dtype=float).reshape(-1, 1)
        y_vec = np.asarray(measurement.y, dtype=float).reshape(-1, 1)
        u_vec = np.asarray(controls[:-1, :], dtype=float).reshape(-1, 1)
        c_vec = np.asarray(contacts, dtype=float).reshape(-1, 1)
        g_vec = DM(g_meas.reshape(-1, 1))
        theta_core_list = np.asarray(theta_core, dtype=float).reshape(-1).tolist()

        kkt_value = self.estimator.KKT_fn(
            s=x_vec,
            n=w_vec,
            costate=lambda_vec,
            y=y_vec,
            u=u_vec,
            c=c_vec,
            prior=prior,
            tp=theta_core_list,
            G=g_vec,
        )["KKT_fn"]
        d_kkt_z = self.estimator.dKKT_Z_fn(
            s=x_vec,
            n=w_vec,
            costate=lambda_vec,
            y=y_vec,
            u=u_vec,
            c=c_vec,
            prior=prior,
            tp=theta_core_list,
            G=g_vec,
        )["dKKT_Z_fn"]
        d_kkt_theta = self.estimator.dKKT_tp_fn(
            s=x_vec,
            n=w_vec,
            costate=lambda_vec,
            y=y_vec,
            u=u_vec,
            c=c_vec,
            prior=prior,
            tp=theta_core_list,
            G=g_vec,
        )["dKKT_tp_fn"]
        d_kkt_y = self.estimator.dKKT_Y_fn(
            s=x_vec,
            n=w_vec,
            costate=lambda_vec,
            y=y_vec,
            u=u_vec,
            c=c_vec,
            prior=prior,
            tp=theta_core_list,
            G=g_vec,
        )["dKKT_Y_fn"]
        d_kkt_g = self.estimator.dKKT_G_fn(
            s=x_vec,
            n=w_vec,
            costate=lambda_vec,
            y=y_vec,
            u=u_vec,
            c=c_vec,
            prior=prior,
            tp=theta_core_list,
            G=g_vec,
        )["dKKT_G"]

        n_system = int(kkt_value.size1())
        fz = _casadi_to_csc(d_kkt_z, (n_system, n_system))
        fy = _casadi_to_csc(d_kkt_y, (n_system, measurement.dy_dtip.shape[0]))
        fg = _casadi_to_csc(d_kkt_g, (n_system, (horizon + 1) * 24 * 9))
        ftheta = np.asarray(d_kkt_theta.full(), dtype=np.float64)

        fg_tip = np.zeros((n_system, 12))
        for k in range(horizon + 1):
            col0 = k * 24 * 9
            fg_k = fg[:, col0 : col0 + 24 * 9]
            dg_k = self.robot.dG_dtip(
                self.codegen.foot_velocity,
                self.codegen.foot_position,
                q_meas[k, :],
                v_meas[k, :],
                u_meas[k, :],
            )
            fg_tip += fg_k @ dg_k

        rhs = np.hstack([ftheta, fy @ measurement.dy_dtip + fg_tip])
        dz_dtheta = -_solve_sparse(fz, rhs)
        dstate_dtheta = dz_dtheta[: (horizon + 1) * n_state, :]

        return SensitivityResult(
            dstate_dtheta=np.asarray(dstate_dtheta, dtype=float),
            kkt_inf_norm=float(np.linalg.norm(kkt_value.full(), np.inf)),
        )
