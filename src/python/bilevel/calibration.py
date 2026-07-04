"""High-level Frank-Wolfe calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from casadi import DM

from .codegen import CodegenFunctions
from .config import BilevelConfig, WeightParameterLayout, default_weight_vector
from .data_io import LeggedDataset
from .estimator.full_information import FullInformationEstimator
from .exports import TrajectoryExporter
from .lmo import LinearMinimizationOracle
from .losses import TrajectoryLoss
from .models import Models
from .robot import B1RobotModel, MeasurementBundle
from .sensitivity import EstimatorSensitivity


@dataclass
class EstimatorSolution:
    state: np.ndarray
    noise: np.ndarray
    costate: np.ndarray


@dataclass
class CalibrationState:
    theta: np.ndarray
    solution: EstimatorSolution
    measurement: MeasurementBundle
    loss: float


@dataclass(frozen=True)
class CalibrationResult:
    theta: np.ndarray
    theta_history: np.ndarray
    loss_history: np.ndarray
    gradient_history: np.ndarray
    state_trajectory: np.ndarray


class FrankWolfeCalibrator:
    """Estimator-in-the-loop calibration for the B1 dataset."""

    def __init__(
        self,
        config: BilevelConfig,
        dataset: LeggedDataset,
        robot: B1RobotModel,
        codegen: CodegenFunctions,
        models: Models | None = None,
        estimator: FullInformationEstimator | None = None,
        loss: TrajectoryLoss | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.window = dataset.window(
            config.dataset.start_idx,
            config.dataset.horizon,
        )
        self.robot = robot
        self.codegen = codegen
        self.models = models or self._build_models()
        self.estimator = estimator or self._build_estimator(self.models)
        self.loss = loss or TrajectoryLoss()
        self.layout = WeightParameterLayout(self.estimator.n_state)
        self.sensitivity = EstimatorSensitivity(self.estimator, self.robot, self.codegen)
        self.lmo = LinearMinimizationOracle(self.layout, config.frank_wolfe)
        self.exporter = TrajectoryExporter(config.output_dir, robot)

        self.prior = self.robot.initial_state_prior(self.window.x[0, :])
        self.g_meas = self.robot.build_measurement_jacobians(
            self.window.q, self.window.v, self.window.u
        )

    def run(self) -> CalibrationResult:
        self.estimator.load_or_build_derivatives(str(self.config.casadi_cache_dir))
        state = self._initial_state()

        max_iter = self.config.frank_wolfe.max_iterations
        theta_history = np.full((max_iter + 1, self.layout.total_size), np.nan)
        loss_history = np.full(max_iter, np.nan)
        gradient_history = np.full(max_iter, np.nan)
        theta_history[0, :] = state.theta

        self._export_iteration(0, state, theta_history)

        for iteration in range(1, max_iter + 1):
            print(f"\n=== Frank-Wolfe iter {iteration} ===")
            print(f"loss = {state.loss:.6g}")
            loss_history[iteration - 1] = state.loss

            gradient, kkt_inf = self._gradient(state)
            gradient_norm = float(np.linalg.norm(gradient))
            gradient_history[iteration - 1] = gradient_norm
            print(f"||KKT||_inf = {kkt_inf:.6g}")
            print(f"||grad|| = {gradient_norm:.6g}")

            lmo_result = self.lmo.solve(gradient, state.theta)
            print(f"LMO status: {lmo_result.status}")

            direction = lmo_result.point - state.theta
            next_state, gamma, expected = self._armijo(state, gradient, direction)
            actual = next_state.loss - state.loss

            theta_history[iteration, :] = next_state.theta
            state = next_state

            print(f"Armijo gamma = {gamma:.3e}")
            print(f"dL_exp = {expected:.6g}")
            print(f"dL = {actual:.6g}")

            self._export_iteration(iteration, state, theta_history)

        self.exporter.save_theta_history(
            theta_history,
            self.layout.core_size,
            self.layout.tip_slice.start,
            self.layout.base_slice.start,
        )
        self.exporter.export_snapshot_csv(
            "end",
            state.solution.state,
            self.window.x,
            self.window.foot,
            self.window.q,
            state.theta[self.layout.base_slice],
            state.theta[self.layout.tip_slice],
        )
        return CalibrationResult(
            theta=state.theta,
            theta_history=theta_history,
            loss_history=loss_history,
            gradient_history=gradient_history,
            state_trajectory=state.solution.state,
        )

    def _build_models(self) -> Models:
        models = Models(self.config.effective_dt)
        models.build_models()
        return models

    def _build_estimator(self, models: Models) -> FullInformationEstimator:
        estimator = FullInformationEstimator(
            self.config.dataset.horizon,
            self.config.effective_dt,
            solver_config=self.config.fatrop,
        )
        estimator.set_state_variable(models.xa)
        estimator.set_output_variable(models.y)
        estimator.set_control_variable(models.u)
        estimator.set_noise_variable(models.w)
        estimator.set_models(models.models_mhe)
        estimator.set_cost_models()
        return estimator

    def _initial_state(self) -> CalibrationState:
        theta_core = np.asarray(default_weight_vector(), dtype=float)
        if theta_core.size != self.layout.core_size:
            raise ValueError(
                f"default weight vector has size {theta_core.size}; "
                f"expected {self.layout.core_size}"
            )
        theta = np.concatenate([theta_core, np.zeros(12), np.zeros(3)])
        measurement = self.robot.build_measurements(
            self.window.q, self.window.v, self.window.u, theta[self.layout.tip_slice]
        )
        solution = self._solve_estimator(measurement, theta_core)
        return CalibrationState(
            theta=theta,
            solution=solution,
            measurement=measurement,
            loss=self._loss_value(solution.state, theta[self.layout.base_slice]),
        )

    def _solve_estimator(
        self, measurement: MeasurementBundle, theta_core: np.ndarray
    ) -> EstimatorSolution:
        opt_sol = self.estimator.solve(
            measurement.y,
            self.window.u,
            self.prior,
            np.asarray(theta_core, dtype=float).reshape(-1).tolist(),
            self.window.horizon,
            self.window.contact,
            self.g_meas,
        )
        return EstimatorSolution(
            state=opt_sol["state_traj_opt"],
            noise=opt_sol["noise_traj_opt"],
            costate=opt_sol["costate"],
        )

    def _loss_value(self, state_traj: np.ndarray, base_offset: np.ndarray) -> float:
        attitude_loss = self._attitude_loss(state_traj)
        return self.loss.evaluate(
            state_traj,
            self.window.x,
            self.window.foot,
            attitude_loss,
            base_offset,
        ).value

    def _gradient(self, state: CalibrationState) -> tuple[np.ndarray, float]:
        sensitivity = self.sensitivity.solve(
            state.solution.state,
            state.solution.noise,
            state.solution.costate,
            state.measurement,
            self.window.u,
            self.window.contact,
            self.prior,
            state.theta[: self.layout.core_size],
            self.g_meas,
            self.window.q,
            self.window.v,
            self.window.u,
        )

        state_mask = self.loss.state_sensitivity_mask(
            self.window.horizon, self.estimator.n_state
        )
        dstate = sensitivity.dstate_dtheta[state_mask, :]
        dloss_dx = self.loss.state_gradient(
            state.solution.state,
            self.window.x,
            self.window.foot,
            self._attitude_gradient(state.solution.state),
            state.theta[self.layout.base_slice],
        )
        core_tip_gradient = (dloss_dx @ dstate).reshape(-1)
        base_gradient = self.loss.base_offset_gradient(
            state.solution.state,
            self.window.x,
            state.theta[self.layout.base_slice],
        )
        return np.concatenate([core_tip_gradient, base_gradient]), sensitivity.kkt_inf_norm

    def _armijo(
        self,
        state: CalibrationState,
        gradient: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[CalibrationState, float, float]:
        gamma = self.config.frank_wolfe.armijo_gamma_init
        linear_model = float(gradient @ direction)
        last_candidate = None

        while True:
            theta_candidate = state.theta + gamma * direction
            measurement = self.robot.build_measurements(
                self.window.q,
                self.window.v,
                self.window.u,
                theta_candidate[self.layout.tip_slice],
            )
            solution = self._solve_estimator(
                measurement, theta_candidate[: self.layout.core_size]
            )
            loss_candidate = self._loss_value(
                solution.state, theta_candidate[self.layout.base_slice]
            )
            candidate = CalibrationState(
                theta=theta_candidate,
                solution=solution,
                measurement=measurement,
                loss=loss_candidate,
            )
            last_candidate = candidate
            rhs = (
                state.loss
                + self.config.frank_wolfe.armijo_rho * gamma * linear_model
            )
            if loss_candidate <= rhs:
                break
            gamma *= self.config.frank_wolfe.armijo_beta
            if gamma < 1e-8:
                print("Armijo reached minimum gamma.")
                break

        return last_candidate, gamma, gamma * linear_model

    def _attitude_gradient(self, state_traj: np.ndarray) -> np.ndarray:
        q_est = state_traj[:, 9:13]
        q_mocap = self.window.x[:, 3:7]
        grad = self.estimator.dL_dQ_fn(
            q=DM(q_est.reshape(-1, 1)),
            qm=DM(q_mocap.reshape(-1, 1)),
        )["dL_dQ"]
        return np.asarray(grad, dtype=float).reshape(self.window.length, 4)

    def _attitude_loss(self, state_traj: np.ndarray) -> float:
        q_est = state_traj[:, 9:13]
        q_mocap = self.window.x[:, 3:7]
        value = self.estimator.L_att_fn(
            q=DM(q_est.reshape(-1, 1)),
            qm=DM(q_mocap.reshape(-1, 1)),
        )["L"]
        return float(value)

    def _export_iteration(
        self,
        iteration: int,
        state: CalibrationState,
        theta_history: np.ndarray,
    ) -> None:
        prefix = f"iter_{iteration:03d}"
        if iteration == 0:
            self.exporter.export_snapshot_csv(
                "start",
                state.solution.state,
                self.window.x,
                self.window.foot,
                self.window.q,
                state.theta[self.layout.base_slice],
                state.theta[self.layout.tip_slice],
            )
        self.exporter.export_iteration_state(
            prefix,
            state.solution.state,
            self.window.q,
            dt=self.config.effective_dt,
        )
        self.exporter.plot_results(
            state.solution.state,
            self.window.x,
            self.window.foot,
            self.window.q,
            "cur",
            state.theta[self.layout.base_slice],
            state.theta[self.layout.tip_slice],
        )
        self.exporter.save_theta_history(
            theta_history,
            self.layout.core_size,
            self.layout.tip_slice.start,
            self.layout.base_slice.start,
        )
        self.exporter.plot_theta_history(
            theta_history,
            self.layout.tip_slice.start,
            self.layout.base_slice.start,
            iteration,
            prefix="cur",
        )
