"""Full-SPD covariance calibration workflow for the contact-aided InEKF."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .invariant_ekf import (
    detach_filter,
    precompute_contact_changes,
    run_rows,
    start_filter,
)

# -----------------------------------------------------------------------------
# constants: covariance groups, initialization, regularization weights

GROUP_ORDER = ["gyro", "accel", "gyro_bias", "accel_bias", "contact_proc", "kin_meas"]
COV_KEY = {
    "gyro": "Qg",
    "accel": "Qa",
    "gyro_bias": "Qbg",
    "accel_bias": "Qba",
    "contact_proc": "Qc",
}
INIT_STD = {
    "gyro": 0.01,
    "accel": 0.3,
    "gyro_bias": 1e-5,
    "accel_bias": 1e-4,
    "contact_proc": 0.1,
    "kin_meas": 0.02,
}
FLOOR = {
    "gyro": 1e-10,
    "accel": 1e-8,
    "gyro_bias": 1e-16,
    "accel_bias": 1e-14,
    "contact_proc": 1e-8,
    "kin_meas": 1e-10,
}
LAMBDA = {"prior": 1e-3, "corr": 1e-3, "cond": 1e-1, "nis": 1e-3}
BIAS_PRIOR_BOOST = 1000.0
MAX_LOG_COND = 6.0

# -----------------------------------------------------------------------------
# config and result containers


@dataclass(frozen=True)
class CalibrationConfig:
    """Training configuration.

    exec_mode "batched" runs all rollouts as one fixed-slot batch with one
    Adam step per synchronized chunk (the fast path); "sequential" is the
    original per-rollout dynamic-dimension loop kept as a reference for
    training-dynamics comparisons. compile_mode None runs the batched step
    eagerly; "default"/"reduce-overhead"/"max-autotune" pass through to
    torch.compile(step, fullgraph=True).
    """

    trim_s: float = 1.0
    s_jitter: float = 1e-12
    epochs: int = 20
    chunk: int = 300
    lr: float = 1e-2
    fallback_lr: float = 3e-3
    bias_lr_factor: float = 1.0 / 30.0
    dtype: torch.dtype = torch.float64
    device: str = "auto"
    require_cuda: bool = False
    exec_mode: str = "batched"
    compile_mode: str | None = "auto"
    profile_stages: bool = False
    seed: int = 0


@dataclass
class Rollout:
    stem: str
    split_label: str
    dt: float
    total_rows: int
    trim0: int
    trim1: int
    imu: torch.Tensor
    p_BC: torch.Tensor
    flags: np.ndarray
    changes: list
    gt_R_WB: torch.Tensor
    gt_v_W: torch.Tensor
    gt_v_B: torch.Tensor
    gt_p_W: torch.Tensor


@dataclass
class TrainingResult:
    modules: "CovarianceModel"
    optimizer: torch.optim.Optimizer
    history: list[dict]
    best: dict
    chunk_trace: list[float]
    runtime_s: float
    lr: float
    final_state_dict: dict
    effective_compile_mode: str
    fallback_reason_code: str | None
    next_epoch: int

# -----------------------------------------------------------------------------
# environment helpers


def project_root(start: Path | None = None) -> Path:
    """Find the project root from a notebook/script working directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "src" / "estimation_calibration_cuda" / "invariant_ekf.py").exists():
            return path
    raise FileNotFoundError("project root not found")


def make_device(require_cuda: bool = True) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the calibration training run")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_kwargs(device: torch.device, dtype: torch.dtype = torch.float64) -> dict:
    return {"device": device, "dtype": dtype}


def assert_cuda_float64(*xs: torch.Tensor) -> None:
    for x in xs:
        if isinstance(x, torch.Tensor):
            assert x.device.type == "cuda", x.device
            assert x.dtype == torch.float64, x.dtype


def seed_everything(seed: int, device: torch.device) -> None:
    """Seed host and Torch generators without touching global dtype."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed_all(seed)


def capture_rng_state(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if device.type == "cuda" else None),
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

# -----------------------------------------------------------------------------
# covariance model: SPD3 parameterization, one module per noise group


class SPD3(torch.nn.Module):
    """3x3 SPD covariance via a scaled Cholesky parameterization."""

    def __init__(
        self,
        init_std,
        *,
        floor: float = 1e-12,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        init_std = torch.as_tensor(init_std, dtype=dtype, device=device).reshape(3)
        raw = torch.zeros(3, 3, dtype=dtype, device=device)
        target_diag = torch.clamp(init_std, min=floor**0.5)
        raw_diag = torch.log(torch.expm1(target_diag))
        raw[0, 0], raw[1, 1], raw[2, 2] = raw_diag
        self.raw_tril = torch.nn.Parameter(raw)
        self.floor = float(floor)
        self.offdiag_scale = float(init_std.mean())

    def L(self) -> torch.Tensor:
        tril = torch.tril(self.raw_tril)
        diag = torch.nn.functional.softplus(torch.diagonal(tril)) + self.floor**0.5
        off = (tril - torch.diag(torch.diagonal(tril))) * self.offdiag_scale
        return off + torch.diag(diag)

    def cov(self) -> torch.Tensor:
        L = self.L()
        return L @ L.T + self.floor * torch.eye(3, dtype=L.dtype, device=L.device)

    def log_eigs(self) -> torch.Tensor:
        return torch.log(torch.linalg.eigvalsh(self.cov()).clamp_min(self.floor))


class CovarianceModel(torch.nn.ModuleDict):
    """One SPD3 per noise group.

    Subclasses ModuleDict so state-dict keys stay flat ("gyro.raw_tril", ...)
    and existing checkpoints load unchanged.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__({
            name: SPD3([INIT_STD[name]] * 3, floor=FLOOR[name], dtype=dtype, device=device)
            for name in GROUP_ORDER
        })

    def covs(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Process covariances keyed Qg/Qa/... plus the kinematic measurement cov."""
        return build_covs(self)

    def param_groups(self, lr: float, bias_lr_factor: float) -> list[dict]:
        """Adam param groups: bias groups get a slower, anchored learning rate."""
        bias = [self[name].raw_tril for name in ("gyro_bias", "accel_bias")]
        main = [self[name].raw_tril for name in GROUP_ORDER
                if name not in ("gyro_bias", "accel_bias")]
        return [
            {"params": main, "lr": lr},
            {"params": bias, "lr": lr * bias_lr_factor},
        ]

    @torch.no_grad()
    def summary(self) -> dict[str, dict]:
        """Per-group eigenvalues, conditioning, and correlation (host floats)."""
        out: dict[str, dict] = {}
        for name in GROUP_ORDER:
            C = self[name].cov().detach()
            eig = torch.linalg.eigvalsh(C)
            d = torch.sqrt(torch.diagonal(C).clamp_min(1e-30))
            corr = (C / (d[:, None] * d[None, :])).cpu().numpy()
            off = corr - np.diag(np.diag(corr))
            out[name] = {
                "eigs": eig.cpu().numpy().tolist(),
                "log_cond": float(torch.log(eig[-1]) - torch.log(eig[0])),
                "max_abs_offdiag_corr": float(np.abs(off).max()),
                "cov": C.cpu().numpy().tolist(),
            }
        return out


def make_cov_modules(
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> CovarianceModel:
    return CovarianceModel(device=device, dtype=dtype)


def build_covs(modules: torch.nn.ModuleDict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    covs = {COV_KEY[name]: modules[name].cov() for name in COV_KEY}
    return covs, modules["kin_meas"].cov()


# -----------------------------------------------------------------------------
# data loading: rollouts, contact schedules, filter seeding


def candidate_reliability(
    p_BC: np.ndarray,
    v_BC: np.ndarray,
    height_scale: float = 0.04,
    speed_scale: float = 0.35,
    floor: float = 1e-3,
) -> np.ndarray:
    z = p_BC[..., 2]
    height_score = np.exp(-(((z - z.min(axis=1, keepdims=True)) / height_scale) ** 2))
    speed_score = np.exp(-((np.linalg.norm(v_BC, axis=-1) / speed_scale) ** 2))
    return np.clip(height_score * speed_score, floor, 1.0)


def hysteresis_contact_schedule(weights: np.ndarray, on: float = 0.5, off: float = 0.05) -> np.ndarray:
    total_rows, n_candidates = weights.shape
    flags = np.zeros((total_rows, n_candidates), dtype=bool)
    state = np.zeros(n_candidates, dtype=bool)
    for k in range(total_rows):
        state = np.where(state, weights[k] >= off, weights[k] >= on)
        flags[k] = state
    return flags


def load_rollout(
    data_root: Path,
    stem: str,
    split_label: str,
    *,
    config: CalibrationConfig,
    device: torch.device,
) -> Rollout:
    dd = tensor_kwargs(device, config.dtype)
    d = np.load(data_root / f"{stem}.npz")
    f = np.load(data_root / f"{stem}.features.npz", allow_pickle=True)
    t = d["meta/sim_time"]
    dt = float(np.median(np.diff(t)))
    p_BC = f["input_kinematics/p_BC"]
    weights = candidate_reliability(p_BC, f["input_kinematics/v_BC"])
    flags = hysteresis_contact_schedule(weights)
    imu = np.concatenate([d["input/imu_gyro"], d["input/imu_accelerometer_native"]], axis=1)
    total_rows = p_BC.shape[0]
    n_trim = int(round(config.trim_s / dt))
    trim0, trim1 = n_trim, total_rows - n_trim
    if trim1 - trim0 <= 1000:
        raise ValueError(f"rollout is too short after trimming: {stem}")
    return Rollout(
        stem=stem,
        split_label=split_label,
        dt=dt,
        total_rows=total_rows,
        trim0=trim0,
        trim1=trim1,
        imu=torch.as_tensor(imu, **dd),
        p_BC=torch.as_tensor(p_BC, **dd),
        flags=flags,
        changes=precompute_contact_changes(flags),
        gt_R_WB=torch.as_tensor(d["gt/gt_R_WB"], **dd),
        gt_v_W=torch.as_tensor(d["gt/base_linear_velocity"], **dd),
        gt_v_B=torch.as_tensor(d["gt/gt_v_B"], **dd),
        gt_p_W=torch.as_tensor(d["gt/base_position"], **dd),
    )


def load_rollouts(
    data_root: Path,
    *,
    config: CalibrationConfig,
    device: torch.device,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[list[str], dict[str, str], dict[str, Rollout]]:
    """Load only requested splits through the validated data boundary."""
    from .data import _episode_to_rollout, load_dataset
    dataset = load_dataset(data_root)
    episodes = [episode for split in splits for episode in dataset.load(split)]
    rollout_order = sorted(episode.name for episode in episodes)
    by_name = {episode.name: episode for episode in episodes}
    split_labels = {name: by_name[name].split for name in rollout_order}
    rollouts = {
        name: _episode_to_rollout(
            by_name[name], device=device, dtype=config.dtype, trim_s=config.trim_s)
        for name in rollout_order
    }
    return rollout_order, split_labels, rollouts


def fixed_initial_covariance(device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    P0 = np.eye(15)
    for sl, scale in [((0, 3), 1e-4), ((3, 6), 1e-2), ((6, 9), 1e-4),
                      ((9, 12), 1e-4), ((12, 15), 1e-2)]:
        P0[sl[0]:sl[1], sl[0]:sl[1]] *= scale
    return torch.as_tensor(P0, device=device, dtype=dtype)


def seed_state(roll: Rollout, row: int, P0_fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached ground-truth seed used only at the trimmed rollout start."""
    X0 = torch.eye(5, dtype=roll.gt_R_WB.dtype, device=roll.gt_R_WB.device)
    X0[0:3, 0:3] = roll.gt_R_WB[row].detach()
    X0[0:3, 3] = roll.gt_v_W[row].detach()
    X0[0:3, 4] = roll.gt_p_W[row].detach()
    theta0 = torch.zeros(6, dtype=roll.gt_R_WB.dtype, device=roll.gt_R_WB.device)
    return X0.detach(), theta0, P0_fixed.detach().clone()

# -----------------------------------------------------------------------------
# replay / eval


def trajectory_metrics(
    R_est: torch.Tensor,
    v_est: torch.Tensor,
    p_est: torch.Tensor,
    gt_R_WB: torch.Tensor,
    gt_v_B: torch.Tensor,
    gt_p_W: torch.Tensor,
    P: torch.Tensor,
    *,
    nis: torch.Tensor | None = None,
    nis_dim: torch.Tensor | None = None,
    jitter_events: int = 0,
) -> dict:
    """Neutral estimator diagnostics with explicit units."""
    v_B = torch.einsum("tji,tj->ti", R_est, v_est)
    body_se = ((v_B - gt_v_B) ** 2).sum(-1)
    position_error = torch.linalg.vector_norm(p_est - gt_p_W, dim=-1)
    relative = R_est.transpose(-1, -2) @ gt_R_WB
    cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    orientation_deg = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    P_sym = 0.5 * (P + P.T)
    corrected = (nis_dim > 0) if nis_dim is not None else None
    corrected_rows = int(corrected.sum()) if corrected is not None else 0
    nis_sum = (float((nis[corrected] / nis_dim[corrected]).sum())
               if corrected_rows else 0.0)
    rows = R_est.shape[0]
    return {
        "body_velocity_rmse_mps": float(torch.sqrt(body_se.mean())),
        "orientation_mean_deg": float(orientation_deg.mean()),
        "orientation_max_deg": float(orientation_deg.max()),
        "position_rmse_m": float(torch.sqrt((position_error ** 2).mean())),
        "position_final_error_m": float(position_error[-1]),
        "nis_per_measurement_dim": nis_sum / corrected_rows if corrected_rows else None,
        "corrected_rows": corrected_rows,
        "rows": int(rows),
        "body_velocity_sse": float(body_se.sum()),
        "position_sse": float((position_error ** 2).sum()),
        "orientation_sum_deg": float(orientation_deg.sum()),
        "nis_per_dim_sum": nis_sum,
        "final_P_sym": float((P - P.T).abs().max()),
        "final_P_min_eig": float(torch.linalg.eigvalsh(P_sym).min()),
        "finite": bool(torch.isfinite(R_est).all() and torch.isfinite(v_est).all()
                       and torch.isfinite(p_est).all() and torch.isfinite(P).all()),
        "jitter_events": int(jitter_events),
    }


def aggregate_metrics(results: dict[str, dict]) -> dict:
    if not results:
        raise ValueError("cannot aggregate an empty evaluation split")
    values = list(results.values())
    rows = sum(value["rows"] for value in values)
    corrected = sum(value["corrected_rows"] for value in values)
    return {
        "body_velocity_rmse_mps": float(np.sqrt(
            sum(value["body_velocity_sse"] for value in values) / rows)),
        "orientation_mean_deg": (
            sum(value["orientation_sum_deg"] for value in values) / rows),
        "orientation_max_deg": max(value["orientation_max_deg"] for value in values),
        "position_rmse_m": float(np.sqrt(
            sum(value["position_sse"] for value in values) / rows)),
        "position_final_error_m": float(np.mean(
            [value["position_final_error_m"] for value in values])),
        "nis_per_measurement_dim": (
            sum(value["nis_per_dim_sum"] for value in values) / corrected
            if corrected else None),
        "corrected_rows": corrected,
        "rows": rows,
        "final_P_sym": max(value["final_P_sym"] for value in values),
        "final_P_min_eig": min(value["final_P_min_eig"] for value in values),
        "finite": all(value["finite"] for value in values),
        "jitter_events": sum(value["jitter_events"] for value in values),
    }


def eval_replay(
    roll: Rollout,
    covs: dict[str, torch.Tensor],
    R_kin: torch.Tensor,
    *,
    P0_fixed: torch.Tensor,
    s_jitter: float = 1e-12,
    return_trajectory: bool = False,
) -> dict:
    s0, s1 = roll.trim0, roll.trim1
    with torch.no_grad():
        X0, theta0, P0 = seed_state(roll, s0, P0_fixed)
        filt = start_filter(X0, theta0, P0, covs, roll.flags[s0], roll.p_BC[s0],
                            R_kin, s_jitter=s_jitter)
        R0 = filt.X[0:3, 0:3].clone()
        v0 = filt.X[0:3, 3].clone()
        p0 = filt.X[0:3, 4].clone()
        out = run_rows(
            filt,
            roll.imu[s0 + 1:s1],
            roll.dt,
            roll.p_BC[s0 + 1:s1],
            None,
            None,
            R_kin,
            collect_nis=True,
            changes_list=roll.changes[s0 + 1:s1],
        )
        R_est = torch.cat([R0[None], out["R_WB"]])
        v_est = torch.cat([v0[None], out["v_W"]])
        p_est = torch.cat([p0[None], out["p_W"]])
        nis = torch.stack(out["nis_values"]) if out["nis_values"] else None
        nis_dim = (torch.as_tensor(out["nis_dims"], device=nis.device, dtype=nis.dtype)
                   if nis is not None else None)
        result = trajectory_metrics(
            R_est, v_est, p_est, roll.gt_R_WB[s0:s1], roll.gt_v_B[s0:s1],
            roll.gt_p_W[s0:s1], filt.P, nis=nis, nis_dim=nis_dim,
            jitter_events=filt.jitter_events,
        )
        if return_trajectory:
            result.update({
                "R_WB": R_est.detach().cpu().numpy(),
                "v_W": v_est.detach().cpu().numpy(),
                "p_W": p_est.detach().cpu().numpy(),
                "rows_slice": (s0, s1),
            })
        return result


def evaluate_all(
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    *,
    covs_initial: dict[str, torch.Tensor],
    R_kin_initial: torch.Tensor,
    covs_calibrated: dict[str, torch.Tensor],
    R_kin_calibrated: torch.Tensor,
    P0_fixed: torch.Tensor,
    s_jitter: float,
) -> dict:
    summary = {"rollouts": {}}
    sse_init = sse_cal = rows_total = 0.0
    for stem in rollout_order:
        roll = rollouts[stem]
        init = eval_replay(roll, covs_initial, R_kin_initial, P0_fixed=P0_fixed,
                           s_jitter=s_jitter)
        cal = eval_replay(roll, covs_calibrated, R_kin_calibrated, P0_fixed=P0_fixed,
                          s_jitter=s_jitter)
        if not (cal["finite"] and cal["final_P_min_eig"] > -1e-12 and cal["final_P_sym"] < 1e-9):
            raise FloatingPointError(f"final covariance check failed for {stem}")
        sse_init += init["body_velocity_sse"]
        sse_cal += cal["body_velocity_sse"]
        rows_total += cal["rows"]
        summary["rollouts"][stem] = {
            "manifest_split_label": roll.split_label,
            "rows": cal["rows"],
            "body_velocity_rmse_initial_mps": init["body_velocity_rmse_mps"],
            "body_velocity_rmse_calibrated_mps": cal["body_velocity_rmse_mps"],
            "orientation_mean_deg": cal["orientation_mean_deg"],
            "orientation_max_deg": cal["orientation_max_deg"],
            "position_rmse_m": cal["position_rmse_m"],
            "position_final_error_m": cal["position_final_error_m"],
            "nis_per_measurement_dim": cal["nis_per_measurement_dim"],
            "final_P_min_eig": cal["final_P_min_eig"],
            "final_P_sym_residual": cal["final_P_sym"],
            "jitter_events": cal["jitter_events"],
        }
    summary["aggregate_body_velocity_rmse_initial_mps"] = float(
        np.sqrt(sse_init / rows_total))
    summary["aggregate_body_velocity_rmse_calibrated_mps"] = float(
        np.sqrt(sse_cal / rows_total))
    return summary

# -----------------------------------------------------------------------------
# regularization


def reg_log_eig_prior(module: SPD3, prior_std: float, *, device: torch.device) -> torch.Tensor:
    target = 2.0 * torch.log(torch.as_tensor(prior_std, dtype=torch.float64, device=device))
    return ((module.log_eigs() - target) ** 2).mean()


def reg_correlation(C: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(torch.diagonal(C).clamp_min(1e-30))
    corr = C / (d[:, None] * d[None, :])
    off = corr - torch.diag(torch.diagonal(corr))
    return (off ** 2).mean()


def reg_condition_number(C: torch.Tensor, max_log_cond: float = MAX_LOG_COND) -> torch.Tensor:
    eig = torch.linalg.eigvalsh(C).clamp_min(1e-30)
    log_cond = torch.log(eig[-1]) - torch.log(eig[0])
    return torch.relu(log_cond - max_log_cond) ** 2


def reg_nis(nis_values: Iterable[torch.Tensor], nis_dims: Iterable[int], *, device: torch.device) -> torch.Tensor:
    nis_values = list(nis_values)
    nis_dims = list(nis_dims)
    if not nis_values:
        return torch.zeros((), dtype=torch.float64, device=device)
    per_dim = torch.stack([n / d for n, d in zip(nis_values, nis_dims)])
    return (per_dim.mean() - 1.0) ** 2


def covariance_regularization(
    modules: torch.nn.ModuleDict,
    nis_values: Iterable[torch.Tensor],
    nis_dims: Iterable[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms: dict[str, torch.Tensor] = {}
    total = torch.zeros((), dtype=torch.float64, device=device)
    for name in GROUP_ORDER:
        C = modules[name].cov()
        boost = BIAS_PRIOR_BOOST if name in ("gyro_bias", "accel_bias") else 1.0
        term = (
            LAMBDA["prior"] * boost * reg_log_eig_prior(modules[name], INIT_STD[name], device=device)
            + LAMBDA["corr"] * reg_correlation(C)
            + LAMBDA["cond"] * reg_condition_number(C)
        )
        terms[name] = term
        total = total + term
    terms["nis"] = LAMBDA["nis"] * reg_nis(nis_values, nis_dims, device=device)
    return total + terms["nis"], terms

# -----------------------------------------------------------------------------
# training: continuous per-rollout replay, chunked BPTT, one Adam step per chunk


def validate_training_splits(
    train_order: list[str],
    train_rollouts: dict[str, Rollout],
    validation_order: list[str],
    validation_rollouts: dict[str, Rollout],
) -> None:
    if not train_order or not validation_order:
        raise ValueError("calibration requires nonempty train and validation splits")
    if len(train_order) != len(set(train_order)) or len(validation_order) != len(
            set(validation_order)):
        raise ValueError("duplicate episode in calibration split")
    if set(train_order) & set(validation_order):
        raise ValueError("train and validation splits overlap")
    if set(train_order) != set(train_rollouts) or set(validation_order) != set(
            validation_rollouts):
        raise ValueError("rollout order does not match split contents")


def _validation_metrics_sequential(
    modules: torch.nn.ModuleDict,
    validation_order: list[str],
    validation_rollouts: dict[str, Rollout],
    *,
    P0_fixed: torch.Tensor,
    s_jitter: float,
) -> dict:
    with torch.no_grad():
        covs, R_kin = build_covs(modules)
        results = {
            name: eval_replay(validation_rollouts[name], covs, R_kin,
                              P0_fixed=P0_fixed, s_jitter=s_jitter)
            for name in validation_order
        }
    return aggregate_metrics(results)


def train_trimmed_rollouts(
    train_order: list[str],
    train_rollouts: dict[str, Rollout],
    *,
    validation_order: list[str],
    validation_rollouts: dict[str, Rollout],
    config: CalibrationConfig,
    device: torch.device,
    validation_callback: Callable[[torch.nn.ModuleDict, int], dict] | None = None,
    resume_state: dict | None = None,
    epoch_callback: Callable[[dict], None] | None = None,
) -> TrainingResult:
    validate_training_splits(train_order, train_rollouts,
                             validation_order, validation_rollouts)
    if resume_state is None:
        seed_everything(config.seed, device)
    modules = make_cov_modules(device=device, dtype=config.dtype)
    params = list(modules.parameters())
    optimizer = torch.optim.Adam(
        modules.param_groups(config.lr, config.bias_lr_factor),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    P0_fixed = fixed_initial_covariance(device, config.dtype)
    if resume_state is None:
        start_epoch = 0
        history: list[dict] = []
        chunk_trace: list[float] = []
        best = {
            "validation_body_velocity_rmse_mps": float("inf"),
            "epoch": -1,
            "state": None,
        }
    else:
        modules.load_state_dict(resume_state["current_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = int(resume_state["next_epoch"])
        history = copy.deepcopy(resume_state["history"])
        chunk_trace = list(resume_state["chunk_trace"])
        best = copy.deepcopy(resume_state["best"])
        restore_rng_state(resume_state["rng_state"], device)
    total_train_rows = sum(r.trim1 - r.trim0 - 1 for r in train_rollouts.values())
    t_train = time.time()
    for epoch in range(start_epoch, config.epochs):
        _reset_peak_memory(device)
        t_epoch = time.time()
        body_losses: list[float] = []
        reg_losses: list[float] = []
        nis_chunk_means: list[torch.Tensor] = []
        grad_norms = {name: [] for name in GROUP_ORDER}
        jitter_events = 0
        for stem in train_order:
            roll = train_rollouts[stem]
            covs, R_kin = build_covs(modules)
            X0, theta0, P0 = seed_state(roll, roll.trim0, P0_fixed)
            if device.type == "cuda":
                assert_cuda_float64(X0, P0, R_kin, *covs.values())
            filt = start_filter(X0, theta0, P0, covs, roll.flags[roll.trim0],
                                roll.p_BC[roll.trim0], R_kin,
                                s_jitter=config.s_jitter)
            a = roll.trim0 + 1
            while a < roll.trim1:
                b = min(a + config.chunk, roll.trim1)
                covs, R_kin = build_covs(modules)
                filt.covs = covs
                out = run_rows(
                    filt,
                    roll.imu[a:b],
                    roll.dt,
                    roll.p_BC[a:b],
                    None,
                    None,
                    R_kin,
                    collect_nis=True,
                    changes_list=roll.changes[a:b],
                )
                v_B = torch.einsum("tji,tj->ti", out["R_WB"], out["v_W"])
                loss_body = ((v_B - roll.gt_v_B[a:b]) ** 2).sum(-1).mean()
                loss_reg, _ = covariance_regularization(
                    modules, out["nis_values"], out["nis_dims"], device=device)
                loss = loss_body + loss_reg
                lb, lr = float(loss_body), float(loss_reg)
                if not (np.isfinite(lb) and np.isfinite(lr)):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    for name in GROUP_ORDER:
                        grad = modules[name].raw_tril.grad
                        grad_norms[name].append(
                            grad.norm() if grad is not None
                            else torch.zeros((), dtype=config.dtype, device=device)
                        )
                    if out["nis_values"]:
                        nis_chunk_means.append(torch.stack(
                            [nv / nd for nv, nd in zip(out["nis_values"], out["nis_dims"])]
                        ).mean())
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                body_losses.append(lb)
                reg_losses.append(lr)
                chunk_trace.append(lb)
                detach_filter(filt)
                a = b
            jitter_events += filt.jitter_events
        rec = {
            "epoch": epoch,
            "train_body_loss": float(np.mean(body_losses)),
            "train_reg_loss": float(np.mean(reg_losses)),
            "nis_per_dim_mean": (float(torch.stack(nis_chunk_means).mean())
                                 if nis_chunk_means else None),
            "jitter_events": jitter_events,
            "peak_gb": _peak_memory_gb(device),
            "epoch_s": time.time() - t_epoch,
            "rows_per_s": total_train_rows / max(time.time() - t_epoch, 1e-12),
            "groups": modules.summary(),
        }
        for name in GROUP_ORDER:
            rec["groups"][name]["grad_norm_mean"] = float(torch.stack(grad_norms[name]).mean())
        with torch.no_grad():
            validation = (validation_callback(modules, epoch)
                          if validation_callback is not None
                          else _validation_metrics_sequential(
                              modules, validation_order, validation_rollouts,
                              P0_fixed=P0_fixed, s_jitter=config.s_jitter))
        metric = float(validation["body_velocity_rmse_mps"])
        if not np.isfinite(metric):
            raise FloatingPointError(f"non-finite validation metric at epoch {epoch}")
        rec["validation"] = validation
        history.append(rec)
        if metric < best["validation_body_velocity_rmse_mps"]:
            best = {
                "validation_body_velocity_rmse_mps": metric,
                "epoch": epoch,
                "state": copy.deepcopy(modules.state_dict()),
            }
        nis_text = (f"{rec['nis_per_dim_mean']:.3f}"
                    if rec["nis_per_dim_mean"] is not None else "n/a")
        print(
            f"epoch {epoch:2d}: body {rec['train_body_loss']:.4f} "
            f"reg {rec['train_reg_loss']:.5f} | val {metric:.4f} m/s "
            f"| NIS/dim {nis_text} "
            f"| {rec['epoch_s']:.0f}s ({rec['rows_per_s']:.0f} rows/s) "
            f"| peak {rec['peak_gb']:.2f} GB"
        )
        if epoch_callback is not None:
            epoch_callback({
                "current_state_dict": modules.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "next_epoch": epoch + 1,
                "history": history,
                "chunk_trace": chunk_trace,
                "best": best,
                "rng_state": capture_rng_state(device),
                "effective_compile_mode": "none",
                "fallback_reason_code": None,
            })
    final_state = copy.deepcopy(modules.state_dict())
    modules.load_state_dict(best["state"])
    return TrainingResult(
        modules=modules,
        optimizer=optimizer,
        history=history,
        best=best,
        chunk_trace=chunk_trace,
        runtime_s=time.time() - t_train,
        lr=config.lr,
        final_state_dict=final_state,
        effective_compile_mode="none",
        fallback_reason_code=None,
        next_epoch=config.epochs,
    )

# -----------------------------------------------------------------------------
# CLI compatibility


def main(argv: list[str] | None = None) -> None:
    from .cli import main as cli_main
    cli_main(argv)


if __name__ == "__main__":
    main()
