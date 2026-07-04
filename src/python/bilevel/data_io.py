"""CSV loading and window slicing for B1 calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BilevelConfig


def read_from_csv(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(arr)
    if factor < 1:
        raise ValueError("downsample factor must be positive")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("downsample input must be a 2D array")
    return arr[::factor]


@dataclass(frozen=True)
class EstimationWindow:
    """A contiguous FIE/FW window with H + 1 samples."""

    start_idx: int
    horizon: int
    y: np.ndarray
    u: np.ndarray
    q: np.ndarray
    v: np.ndarray
    x: np.ndarray
    foot: np.ndarray
    contact: np.ndarray

    @property
    def length(self) -> int:
        return self.horizon + 1

    @property
    def controls_for_transitions(self) -> np.ndarray:
        return self.u[:-1, :]


@dataclass(frozen=True)
class LeggedDataset:
    """Full loaded dataset."""

    y: np.ndarray
    u: np.ndarray
    q: np.ndarray
    v: np.ndarray
    x: np.ndarray
    foot: np.ndarray
    contact: np.ndarray

    def window(self, start_idx: int, horizon: int) -> EstimationWindow:
        end = start_idx + horizon + 1
        n_rows = self.x.shape[0]
        if start_idx < 0 or end > n_rows:
            raise ValueError(
                f"window [{start_idx}, {end}) exceeds dataset length {n_rows}"
            )
        return EstimationWindow(
            start_idx=start_idx,
            horizon=horizon,
            y=self.y[start_idx:end, :],
            u=self.u[start_idx:end, :],
            q=self.q[start_idx:end, :],
            v=self.v[start_idx:end, :],
            x=self.x[start_idx:end, :],
            foot=self.foot[start_idx:end, :],
            contact=self.contact[start_idx:end, :],
        )


class CsvDatasetLoader:
    """Loads the repository CSV files and applies reference preprocessing."""

    def __init__(self, config: BilevelConfig):
        self.config = config

    def load(self) -> LeggedDataset:
        data_dir = Path(self.config.data_dir)
        ds = self.config.dataset.downsample_factor

        y_data = downsample(read_from_csv(data_dir / "y_mocap.csv"), ds)
        u_data = downsample(read_from_csv(data_dir / "u_mocap.csv"), ds)
        q_data = downsample(read_from_csv(data_dir / "q_mocap.csv"), ds)
        v_data = downsample(read_from_csv(data_dir / "v_mocap.csv"), ds)
        x_data = downsample(read_from_csv(data_dir / "x_mocap.csv"), ds)
        foot_data = downsample(read_from_csv(data_dir / "foot_mocap.csv"), ds)
        contact_data = downsample(read_from_csv(data_dir / "contact_mocap.csv"), ds)

        foot_data = foot_data.copy()
        for leg_idx, z_offset in enumerate(self.config.dataset.foot_z_offsets):
            foot_data[:, 3 * leg_idx + 2] += float(z_offset)

        contact_data = (
            contact_data >= self.config.dataset.contact_threshold
        ).astype(float)

        return LeggedDataset(
            y=y_data,
            u=u_data,
            q=q_data,
            v=v_data,
            x=x_data,
            foot=foot_data,
            contact=contact_data,
        )


def load_dataset(cfg: BilevelConfig) -> LeggedDataset:
    return CsvDatasetLoader(cfg).load()
