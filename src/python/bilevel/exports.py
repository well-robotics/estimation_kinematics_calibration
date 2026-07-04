"""CSV and plot export helpers for Frank-Wolfe calibration runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


def _quat_to_rpy_batch(q_arr: np.ndarray) -> np.ndarray:
    rpy = np.zeros((q_arr.shape[0], 3), dtype=float)
    for k, quat in enumerate(q_arr):
        quat = np.asarray(quat, dtype=float)
        quat = quat / (np.linalg.norm(quat) + 1e-16)
        rpy[k, :] = pin.rpy.matrixToRpy(pin.Quaternion(quat).toRotationMatrix())
    return rpy


def _world_velocity_from_mocap(x_window: np.ndarray) -> np.ndarray:
    q_gt = np.asarray(x_window[:, 3:7], dtype=float)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)
    velocity_body = np.asarray(x_window[:, 19:22], dtype=float)
    velocity_world = np.zeros_like(velocity_body)
    for k in range(q_gt.shape[0]):
        velocity_world[k] = pin.Quaternion(q_gt[k]).toRotationMatrix() @ velocity_body[k]
    return velocity_world


def _base_offset_world(x_window: np.ndarray, base_offset: np.ndarray) -> np.ndarray:
    q_gt = np.asarray(x_window[:, 3:7], dtype=float)
    q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)
    base_offset = np.asarray(base_offset, dtype=float).reshape(3)
    return np.vstack(
        [pin.Quaternion(q_gt[k]).toRotationMatrix() @ base_offset for k in range(q_gt.shape[0])]
    )


class TrajectoryExporter:
    def __init__(self, output_dir: str | Path, robot=None):
        self.output_dir = Path(output_dir)
        self.robot = robot
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_results(
        self,
        state_traj: np.ndarray,
        x_window: np.ndarray,
        foot_window: np.ndarray,
        q_meas_window: np.ndarray,
        prefix: str,
        base_offset: np.ndarray | None = None,
        tip_offset: np.ndarray | None = None,
    ) -> None:
        t = np.arange(x_window.shape[0])
        p_est = np.asarray(state_traj[:, 0:3], dtype=float)
        v_est = np.asarray(state_traj[:, 3:6], dtype=float)
        q_est = np.asarray(state_traj[:, 9:13], dtype=float)
        foot_est = np.asarray(state_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)

        p_gt = np.asarray(x_window[:, 0:3], dtype=float)
        q_gt = np.asarray(x_window[:, 3:7], dtype=float)
        q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)
        v_gt = _world_velocity_from_mocap(x_window)
        foot_gt = np.asarray(foot_window, dtype=float).reshape(-1, 4, 3)

        if base_offset is None:
            base_offset = np.zeros(3)
        p_plot = p_est + _base_offset_world(x_window, base_offset)

        foot_plot = foot_est.copy()
        if (
            tip_offset is not None
            and self.robot is not None
            and q_meas_window is not None
        ):
            foot_plot += self.robot.rotated_tip_offsets(q_meas_window, tip_offset)

        axes = ("x", "y", "z")
        fig1, axs1 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, axis in enumerate(axes):
            axs1[i].plot(t, p_plot[:, i], "-", label="est")
            axs1[i].plot(t, p_gt[:, i], "--", label="gt")
            axs1[i].set_ylabel(f"p_{axis}")
            axs1[i].grid(True)
            axs1[i].legend()
        axs1[-1].set_xlabel("t")
        fig1.suptitle("position")
        fig1.tight_layout()

        fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, axis in enumerate(axes):
            axs2[i].plot(t, v_est[:, i], "-", label="est")
            axs2[i].plot(t, v_gt[:, i], "--", label="gt")
            axs2[i].set_ylabel(f"v_{axis}")
            axs2[i].grid(True)
            axs2[i].legend()
        axs2[-1].set_xlabel("t")
        fig2.suptitle("velocity")
        fig2.tight_layout()

        fig3, axs3 = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        for i, label in enumerate(("qx", "qy", "qz", "qw")):
            axs3[i].plot(t, q_est[:, i], "-", label="est")
            axs3[i].plot(t, q_gt[:, i], "--", label="gt")
            axs3[i].set_ylabel(label)
            axs3[i].grid(True)
            axs3[i].legend()
        axs3[-1].set_xlabel("t")
        fig3.suptitle("quaternion")
        fig3.tight_layout()

        fig4, axs4 = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
        for leg_idx, leg in enumerate(("FR", "FL", "RR", "RL")):
            for axis_idx, axis in enumerate(axes):
                ax = axs4[leg_idx, axis_idx]
                ax.plot(t, foot_plot[:, leg_idx, axis_idx], "-", label="est")
                ax.plot(t, foot_gt[:, leg_idx, axis_idx], "--", label="mocap")
                if leg_idx == 0:
                    ax.set_title(axis)
                if axis_idx == 0:
                    ax.set_ylabel(leg)
                if leg_idx == 3:
                    ax.set_xlabel("t")
                if leg_idx == 0 and axis_idx == 0:
                    ax.legend()
                ax.grid(True)
        fig4.suptitle("feet")
        fig4.tight_layout()

        fig1.savefig(self.output_dir / f"{prefix}_pos.png", dpi=200, bbox_inches="tight")
        fig2.savefig(self.output_dir / f"{prefix}_vel.png", dpi=200, bbox_inches="tight")
        fig3.savefig(self.output_dir / f"{prefix}_quat.png", dpi=200, bbox_inches="tight")
        fig4.savefig(self.output_dir / f"{prefix}_feet.png", dpi=200, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)

    def export_snapshot_csv(
        self,
        prefix: str,
        state_traj: np.ndarray,
        x_window: np.ndarray,
        foot_window: np.ndarray,
        q_meas_window: np.ndarray,
        base_offset: np.ndarray | None = None,
        tip_offset: np.ndarray | None = None,
    ) -> None:
        t = np.arange(x_window.shape[0])
        p_est = np.asarray(state_traj[:, 0:3], dtype=float)
        v_est = np.asarray(state_traj[:, 3:6], dtype=float)
        q_est = np.asarray(state_traj[:, 9:13], dtype=float)
        foot_est = np.asarray(state_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)

        p_gt = np.asarray(x_window[:, 0:3], dtype=float)
        q_gt = np.asarray(x_window[:, 3:7], dtype=float)
        q_gt = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-16)
        v_gt = _world_velocity_from_mocap(x_window)
        foot_gt = np.asarray(foot_window, dtype=float).reshape(-1, 4, 3)

        if base_offset is None:
            base_offset = np.zeros(3)
        p_plot = p_est + _base_offset_world(x_window, base_offset)

        foot_plot = foot_est.copy()
        if (
            tip_offset is not None
            and self.robot is not None
            and q_meas_window is not None
        ):
            foot_plot += self.robot.rotated_tip_offsets(q_meas_window, tip_offset)

        np.savetxt(
            self.output_dir / f"{prefix}_pos.csv",
            np.column_stack([t, p_plot, p_gt]),
            delimiter=",",
            header="t,px_est,py_est,pz_est,px_gt,py_gt,pz_gt",
            comments="",
        )
        np.savetxt(
            self.output_dir / f"{prefix}_vel.csv",
            np.column_stack([t, v_est, v_gt]),
            delimiter=",",
            header="t,vx_est,vy_est,vz_est,vx_gt,vy_gt,vz_gt",
            comments="",
        )

        cols = [t]
        headers = ["t"]
        for leg_idx, leg in enumerate(("FR", "FL", "RR", "RL")):
            for axis_idx, axis in enumerate(("x", "y", "z")):
                cols += [foot_plot[:, leg_idx, axis_idx], foot_gt[:, leg_idx, axis_idx]]
                headers += [f"{leg}_{axis}_est", f"{leg}_{axis}_gt"]
        np.savetxt(
            self.output_dir / f"{prefix}_feet.csv",
            np.column_stack(cols),
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

        np.savetxt(
            self.output_dir / f"{prefix}_rpy.csv",
            np.column_stack([t, _quat_to_rpy_batch(q_est), _quat_to_rpy_batch(q_gt)]),
            delimiter=",",
            header="t,roll_est,pitch_est,yaw_est,roll_gt,pitch_gt,yaw_gt",
            comments="",
        )

    def export_iteration_state(
        self,
        prefix: str,
        state_traj: np.ndarray,
        q_meas_window: np.ndarray,
        dt: float | None = None,
    ) -> None:
        h1 = state_traj.shape[0]
        if q_meas_window.shape[0] != h1:
            raise ValueError("q_meas_window rows must equal state trajectory rows")

        t_idx = np.arange(h1).reshape(-1, 1)
        cols = [t_idx]
        headers = ["t"]
        if dt is not None:
            cols.append(t_idx * float(dt))
            headers.append("t_sec")

        cols += [
            np.asarray(state_traj[:, 0:3], dtype=float),
            np.asarray(state_traj[:, 9:13], dtype=float),
            np.asarray(state_traj[:, 3:6], dtype=float),
        ]
        headers += [
            "p_base_x", "p_base_y", "p_base_z",
            "q_base_x", "q_base_y", "q_base_z", "q_base_w",
            "v_base_x", "v_base_y", "v_base_z",
        ]

        n_joint = q_meas_window.shape[1] - 7
        if n_joint > 0:
            cols.append(np.asarray(q_meas_window[:, 7 : 7 + n_joint], dtype=float))
            headers += [f"joint_pos_{i}" for i in range(n_joint)]

        out_dir = self.output_dir / "iterations"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_dir / f"{prefix}_state.csv",
            np.column_stack(cols),
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

    def save_theta_history(
        self,
        theta_history: np.ndarray,
        core_size: int,
        tip_start: int,
        base_start: int,
        filename: str = "theta_history.csv",
    ) -> None:
        valid_rows = np.where(~np.isnan(theta_history[:, 0]))[0]
        if valid_rows.size == 0:
            return
        last_iter = int(valid_rows[-1])
        hist = theta_history[: last_iter + 1, :]
        iter_col = np.arange(last_iter + 1).reshape(-1, 1)

        headers = ["iter"]
        headers += [f"core_{i}" for i in range(core_size)]
        for leg in ("FR", "FL", "RR", "RL"):
            for axis in ("x", "y", "z"):
                headers.append(f"tip_{leg}_{axis}")
        headers += ["base_x", "base_y", "base_z"]

        mat = np.column_stack(
            [
                iter_col,
                hist[:, :core_size],
                hist[:, tip_start : tip_start + 12],
                hist[:, base_start : base_start + 3],
            ]
        )
        np.savetxt(
            self.output_dir / filename,
            mat,
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

    def plot_theta_history(
        self,
        theta_history: np.ndarray,
        tip_start: int,
        base_start: int,
        upto_iter: int,
        prefix: str = "cur",
    ) -> None:
        iterations = np.arange(upto_iter + 1)
        tip_hist = theta_history[: upto_iter + 1, tip_start : tip_start + 12]
        base_hist = theta_history[: upto_iter + 1, base_start : base_start + 3]

        fig1, axs = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
        labels = [
            "FR_x", "FR_y", "FR_z", "FL_x", "FL_y", "FL_z",
            "RR_x", "RR_y", "RR_z", "RL_x", "RL_y", "RL_z",
        ]
        for j, label in enumerate(labels):
            row, col = divmod(j, 3)
            axs[row, col].plot(iterations, tip_hist[:, j], "-", linewidth=1.5)
            axs[row, col].set_ylabel(label)
            axs[row, col].grid(True)
            if row == 3:
                axs[row, col].set_xlabel("Iteration")
        fig1.suptitle("Tip offsets")
        fig1.tight_layout()

        fig2, axs2 = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i, label in enumerate(("base_x", "base_y", "base_z")):
            axs2[i].plot(iterations, base_hist[:, i], "-", linewidth=1.5)
            axs2[i].set_ylabel(label)
            axs2[i].grid(True)
        axs2[-1].set_xlabel("Iteration")
        fig2.suptitle("Base offset")
        fig2.tight_layout()

        fig1.savefig(self.output_dir / f"{prefix}_theta_tip.png", dpi=200, bbox_inches="tight")
        fig2.savefig(self.output_dir / f"{prefix}_theta_base.png", dpi=200, bbox_inches="tight")
        plt.close(fig1)
        plt.close(fig2)
