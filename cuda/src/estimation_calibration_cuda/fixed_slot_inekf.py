"""Fixed-slot, static-shape, batched differentiable contact-aided InEKF.

Same math as ``invariant_ekf.InvariantEKF`` (the parity oracle), restructured
for GPU execution: all 8 contact candidate slots are always materialized
(X: (B, 13, 13), P: (B, 39, 39)), contact insertion/removal becomes masked
tensor ops driven by a schedule precomputed on the host, and every step is a
pure tensor->tensor function with static shapes and no data-dependent Python
control flow -- so it batches over rollouts and is torch.compile / CUDA-graph
friendly.

Equivalence to the dynamic-dimension filter (verified by tests/test_parity.py):
- Inactive slots never leak into active blocks: no row of A reads contact
  columns, H columns at inactive slots are zero (so K columns are exactly 0),
  and insertion overwrites the slot's full P row/column.
- Removal is a pure mask clear; insertion reproduces F P F^T + G cov G^T via
  row copy -> column copy (of the row-updated matrix) -> masked diagonal add,
  which also gets same-row multi-insertion cross terms right.
- Rows with no active measurement are exact no-ops (K = 0, exp(0) = I), and
  dt = 0 rows are bitwise no-op propagates -- used for batch padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import torch

N_SLOTS = 8
DIM_X = 5 + N_SLOTS            # 13
DIM_THETA = 6
GROUP = 3 * DIM_X - 6          # 33: rotation/velocity/position + slot errors
DIM_P = GROUP + DIM_THETA      # 39
DIM_M = 3 * N_SLOTS            # 24 stacked measurement rows

_SMALL_ANGLE = 1e-4
_CONST_CACHE: dict[tuple, torch.Tensor] = {}


def _const(name: str, dtype, device) -> torch.Tensor:
    """Cached read-only constant tensors (identities, H template, skew(g))."""
    key = (name, dtype, str(device))
    t = _CONST_CACHE.get(key)
    if t is None:
        if name == "I3":
            t = torch.eye(3, dtype=dtype, device=device)
        elif name == "I13":
            t = torch.eye(DIM_X, dtype=dtype, device=device)
        elif name == "I24":
            t = torch.eye(DIM_M, dtype=dtype, device=device)
        elif name == "I39":
            t = torch.eye(DIM_P, dtype=dtype, device=device)
        elif name == "H":
            # slot m rows 3m:3m+3: -I at position error (6:9), +I at slot error
            t = torch.zeros(DIM_M, DIM_P, dtype=dtype, device=device)
            I3 = _const("I3", dtype, device)
            for m in range(N_SLOTS):
                t[3 * m:3 * m + 3, 6:9] = -I3
                t[3 * m:3 * m + 3, 9 + 3 * m:12 + 3 * m] = I3
        elif name == "skew_g":
            t = skew(torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device))
        elif name == "g":
            t = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        else:
            raise KeyError(name)
        _CONST_CACHE[key] = t
    return t


def skew(v: torch.Tensor) -> torch.Tensor:
    """Batched skew: (..., 3) -> (..., 3, 3). Matches invariant_ekf.skew."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def _so3_coefficients(theta: torch.Tensor):
    """(sin t/t, (1-cos t)/t^2, (t-sin t)/t^3) with the same small-angle
    series as invariant_ekf._so3_coefficients (STABLE_TRAINING mode)."""
    small = theta < _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    t2 = safe * safe
    a = torch.where(small, 1.0 - theta * theta / 6.0, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta * theta / 24.0, (1.0 - torch.cos(safe)) / t2)
    c = torch.where(small, 1.0 / 6.0 - theta * theta / 120.0,
                    (safe - torch.sin(safe)) / (t2 * safe))
    return a, b, c


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) exp: (B, 3) -> (B, 3, 3)."""
    A = skew(w)
    theta = torch.linalg.norm(w, dim=-1)
    a, b, _ = _so3_coefficients(theta)
    I = _const("I3", w.dtype, w.device)
    return I + a[:, None, None] * A + b[:, None, None] * (A @ A)


def exp_sek3(xi: torch.Tensor) -> torch.Tensor:
    """Batched SE_K(3) exp at fixed K = DIM_X - 3: (B, 33) -> (B, 13, 13)."""
    B = xi.shape[0]
    w = xi[:, :3]
    A = skew(w)
    theta = torch.linalg.norm(w, dim=-1)
    a, b, c = _so3_coefficients(theta)
    I = _const("I3", xi.dtype, xi.device)
    A2 = A @ A
    R = I + a[:, None, None] * A + b[:, None, None] * A2
    Jl = I + b[:, None, None] * A + c[:, None, None] * A2
    X = _const("I13", xi.dtype, xi.device).expand(B, -1, -1).clone()
    X[:, 0:3, 0:3] = R
    cols = torch.einsum("bij,bsj->bis", Jl, xi[:, 3:].reshape(B, DIM_X - 3, 3))
    X[:, 0:3, 3:DIM_X] = cols
    return X


def _sym(P: torch.Tensor) -> torch.Tensor:
    return 0.5 * (P + P.transpose(-1, -2))


def _slot_blockdiag(blocks: torch.Tensor) -> torch.Tensor:
    """(B, 8, 3, 3) slot blocks -> (B, 24, 24) block-diagonal."""
    B = blocks.shape[0]
    out = torch.zeros(B, N_SLOTS, 3, N_SLOTS, 3,
                      dtype=blocks.dtype, device=blocks.device)
    idx = torch.arange(N_SLOTS, device=blocks.device)
    out[:, idx, :, idx, :] = blocks.transpose(0, 1)
    return out.reshape(B, DIM_M, DIM_M)


# -----------------------------------------------------------------------------
# state and schedule containers


class State(NamedTuple):
    X: torch.Tensor            # (B, 13, 13)
    theta: torch.Tensor        # (B, 6)
    P: torch.Tensor            # (B, 39, 39)
    jitter_count: torch.Tensor  # (B,) on-device near-singularity counter
    info_count: torch.Tensor    # (B,) on-device cholesky info != 0 counter


class RowOut(NamedTuple):
    R: torch.Tensor            # (B, 3, 3)
    v: torch.Tensor            # (B, 3)
    p: torch.Tensor            # (B, 3)
    nis: torch.Tensor          # (B,)


@dataclass
class BatchData:
    """Padded batch of rollout segments with the precomputed slot schedule.

    Local row t maps to global row trim0 + t; row 0 is the seed row
    (insert-only, ``apply_row0``), rows >= 1 are filter steps. Padded rows
    have dt_row = 0 and all-false masks, which makes the step an exact no-op.
    """
    B: int
    T_pad: int
    imu: torch.Tensor           # (B, T, 6)
    p_meas: torch.Tensor        # (B, T, 8, 3)
    gt_v_B: torch.Tensor        # (B, T, 3)
    dt_row: torch.Tensor        # (B, T)
    valid: torch.Tensor         # (B, T) bool
    prop_mask: torch.Tensor     # (B, T, 8) bool: active during propagate (flags[k-1])
    correct_mask: torch.Tensor  # (B, T, 8) bool: flags[k-1] & flags[k]
    insert_mask: torch.Tensor   # (B, T, 8) bool: ~flags[k-1] & flags[k]
    nis_dim: torch.Tensor       # (B, T) float: 3 * n corrected slots (0 => no meas)


def build_batch(rolls, *, T_pad: int | None = None,
                dtype: torch.dtype = torch.float64) -> BatchData:
    """Stack trimmed rollout segments into a padded BatchData.

    ``rolls`` are covariance_calibration.Rollout-like objects (imu, p_BC,
    flags, dt, trim0, trim1, gt_v_B attributes).
    """
    device = rolls[0].imu.device
    segs = [(r.trim0, r.trim1 - r.trim0) for r in rolls]
    T = max(s[1] for s in segs) if T_pad is None else T_pad
    B = len(rolls)
    imu = torch.zeros(B, T, 6, dtype=dtype, device=device)
    p_meas = torch.zeros(B, T, N_SLOTS, 3, dtype=dtype, device=device)
    gt_v_B = torch.zeros(B, T, 3, dtype=dtype, device=device)
    dt_row = torch.zeros(B, T, dtype=dtype, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    prop = np.zeros((B, T, N_SLOTS), dtype=bool)
    corr = np.zeros((B, T, N_SLOTS), dtype=bool)
    ins = np.zeros((B, T, N_SLOTS), dtype=bool)
    for b, (r, (t0, L)) in enumerate(zip(rolls, segs)):
        imu[b, :L] = r.imu[t0:t0 + L].to(dtype)
        p_meas[b, :L] = r.p_BC[t0:t0 + L].to(dtype)
        gt_v_B[b, :L] = r.gt_v_B[t0:t0 + L].to(dtype)
        dt_row[b, 1:L] = float(r.dt)
        valid[b, :L] = True
        flags = np.asarray(r.flags[t0:t0 + L]).astype(bool)
        prev = np.zeros_like(flags)
        prev[1:] = flags[:-1]           # prev[0] = 0: row 0 is insert-only
        prop[b, :L] = prev
        corr[b, :L] = prev & flags
        ins[b, :L] = ~prev & flags
    to_t = lambda a: torch.as_tensor(a, device=device)
    corr_t = to_t(corr)
    return BatchData(
        B=B, T_pad=T, imu=imu, p_meas=p_meas, gt_v_B=gt_v_B,
        dt_row=dt_row, valid=valid,
        prop_mask=to_t(prop), correct_mask=corr_t, insert_mask=to_t(ins),
        nis_dim=3.0 * corr_t.sum(-1).to(dtype),
    )


def init_state(seeds, *, device, dtype: torch.dtype = torch.float64) -> State:
    """Embed (X0 5x5, theta0 6, P0 15x15) seeds into fixed-slot state."""
    B = len(seeds)
    X = _const("I13", dtype, device).expand(B, -1, -1).clone()
    theta = torch.zeros(B, DIM_THETA, dtype=dtype, device=device)
    P = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    for b, (X0, theta0, P0) in enumerate(seeds):
        X[b, 0:5, 0:5] = X0.to(dtype)
        theta[b] = theta0.to(dtype)
        P[b, 0:9, 0:9] = P0[0:9, 0:9].to(dtype)
        P[b, 0:9, GROUP:] = P0[0:9, 9:15].to(dtype)
        P[b, GROUP:, 0:9] = P0[9:15, 0:9].to(dtype)
        P[b, GROUP:, GROUP:] = P0[9:15, 9:15].to(dtype)
    zero = torch.zeros(B, dtype=dtype, device=device)
    return State(X, theta, P, zero, zero.clone())


def detach_state(state: State) -> State:
    return State(*(t.detach() for t in state))


# -----------------------------------------------------------------------------
# filter stages (all batched, static shapes)


def _propagate(state: State, gyro, accel, dt_row, prop_mask, covs) -> State:
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_old = X[:, 0:3, 0:3]
    v_old = X[:, 0:3, 3]
    p_old = X[:, 0:3, 4]
    dt = dt_row[:, None]

    w = gyro - theta[:, 0:3]
    a = accel - theta[:, 3:6]
    R_pred = R_old @ exp_so3(w * dt)
    acc_w = torch.einsum("bij,bj->bi", R_old, a) + _const("g", dtype, device)
    v_pred = v_old + acc_w * dt
    p_pred = p_old + v_old * dt + 0.5 * acc_w * dt * dt

    X_new = X.clone()
    X_new[:, 0:3, 0:3] = R_pred
    X_new[:, 0:3, 3] = v_pred
    X_new[:, 0:3, 4] = p_pred

    # error-state Jacobian A (mirrors the dynamic build, all slots materialized)
    cols = X[:, 0:3, 3:DIM_X]                        # v, p, slot columns
    SR = skew(cols.transpose(1, 2)) @ R_old[:, None]  # (B, 10, 3, 3)
    A = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    A[:, 3:6, 0:3] = _const("skew_g", dtype, device)
    A[:, 6:9, 3:6] = _const("I3", dtype, device)
    A[:, 0:3, GROUP:GROUP + 3] = -R_old
    A[:, 3:6, GROUP + 3:] = -R_old
    A[:, 3:GROUP, GROUP:GROUP + 3] = -SR.reshape(B, GROUP - 3, 3)

    # process noise: Qc only on slots active during this interval
    Qk = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    Qk[:, 0:3, 0:3] = covs["Qg"]
    Qk[:, 3:6, 3:6] = covs["Qa"]
    slot_q = covs["Qc"] * prop_mask.to(dtype)[:, :, None, None]
    Qk[:, 9:GROUP, 9:GROUP] = _slot_blockdiag(slot_q)
    Qk[:, GROUP:GROUP + 3, GROUP:GROUP + 3] = covs["Qbg"]
    Qk[:, GROUP + 3:, GROUP + 3:] = covs["Qba"]

    I39 = _const("I39", dtype, device)
    Phi = I39 + A * dt[:, :, None]
    Adj = I39.expand(B, -1, -1).clone()
    Adj[:, 0:3, 0:3] = R_old
    Adj[:, 3:GROUP, 0:3] = SR.reshape(B, GROUP - 3, 3)
    R_diag = R_old[:, None].expand(B, DIM_X - 3, 3, 3)
    Adj[:, 3:GROUP, 3:GROUP] = _slot_blockdiag_k(R_diag)
    PhiAdj = Phi @ Adj
    Qk_hat = PhiAdj @ Qk @ PhiAdj.transpose(1, 2) * dt[:, :, None]
    P_new = _sym(Phi @ P @ Phi.transpose(1, 2) + Qk_hat)
    return State(X_new, theta, P_new, state.jitter_count, state.info_count)


def _slot_blockdiag_k(blocks: torch.Tensor) -> torch.Tensor:
    """(B, K, 3, 3) -> (B, 3K, 3K) block diagonal (K = DIM_X - 3)."""
    B, K = blocks.shape[0], blocks.shape[1]
    out = torch.zeros(B, K, 3, K, 3, dtype=blocks.dtype, device=blocks.device)
    idx = torch.arange(K, device=blocks.device)
    out[:, idx, :, idx, :] = blocks.transpose(0, 1)
    return out.reshape(B, 3 * K, 3 * K)


def _correct(state: State, p_meas, correct_mask, R_kin,
             s_jitter: float) -> tuple[State, torch.Tensor, torch.Tensor]:
    """Masked stacked kinematic correction. Returns (state, nis, N_blk) where
    N_blk = R_pre R_kin R_pre^T is reused by the insertion stage."""
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_pre = X[:, 0:3, 0:3]
    m = correct_mask.to(dtype)                       # (B, 8)
    mrow = m.repeat_interleave(3, dim=1)             # (B, 24)

    H = _const("H", dtype, device) * mrow[:, :, None]
    N_blk = R_pre @ R_kin @ R_pre.transpose(1, 2)    # (B, 3, 3)
    I3 = _const("I3", dtype, device)
    slot_N = torch.where(correct_mask[:, :, None, None],
                         N_blk[:, None], I3.expand(B, N_SLOTS, 3, 3))
    N = _slot_blockdiag(slot_N)

    # innovation rows 0:3 per slot: R p_bc + p - X[0:3, slot], masked
    Z = (torch.einsum("bik,bsk->bsi", R_pre, p_meas)
         + X[:, 0:3, 4][:, None] - X[:, 0:3, 5:DIM_X].transpose(1, 2))
    Z = (Z * m[:, :, None]).reshape(B, DIM_M)

    PHT = P @ H.transpose(1, 2)                      # (B, 39, 24)
    S = H @ PHT + N
    jitter_count = state.jitter_count
    if s_jitter > 0.0:
        jitter_count = jitter_count + (
            torch.diagonal(S, dim1=1, dim2=2).min(dim=1).values
            < 10.0 * s_jitter).to(dtype).detach()
        S = S + s_jitter * _const("I24", dtype, device)
    L, info = torch.linalg.cholesky_ex(S, check_errors=False)
    K = torch.cholesky_solve(PHT.transpose(1, 2), L).transpose(1, 2)
    info_count = state.info_count + (info != 0).to(dtype).detach()

    delta = torch.einsum("bij,bj->bi", K, Z)
    dX = exp_sek3(delta[:, :GROUP])
    X_new = dX @ X
    theta_new = theta + delta[:, GROUP:]
    IKH = _const("I39", dtype, device) - K @ H
    P_new = _sym(IKH @ P @ IKH.transpose(1, 2)
                 + K @ N @ K.transpose(1, 2))
    nis = (Z[:, :, None] * torch.cholesky_solve(Z[:, :, None], L)).sum((1, 2))
    return State(X_new, theta_new, P_new, jitter_count, info_count), nis, N_blk


def _insert(state: State, p_meas, insert_mask, R_pre, N_blk) -> State:
    """Masked slot insertion: PRE-correction R with POST-correction p and P
    (the Hartley augmentation convention). Exact per-slot equivalent of
    sequential F P F^T + G cov G^T, including same-row multi-insert cross
    terms (row copy -> column copy of row-updated -> masked diagonal add)."""
    X, P = state.X, state.P
    B = X.shape[0]
    dtype = X.dtype
    m = insert_mask[:, :, None]                       # (B, 8, 1)
    p_post = X[:, 0:3, 4]

    new_cols = p_post[:, None] + torch.einsum("bik,bsk->bsi", R_pre, p_meas)
    X_new = X.clone()
    X_new[:, 0:3, 5:DIM_X] = torch.where(
        m.transpose(1, 2), new_cols.transpose(1, 2), X[:, 0:3, 5:DIM_X])

    mr = insert_mask[:, :, None, None]                # (B, 8, 1, 1)
    P1 = P.clone()
    rows = P[:, 9:GROUP].reshape(B, N_SLOTS, 3, DIM_P)
    P1[:, 9:GROUP] = torch.where(
        mr, P[:, 6:9, :][:, None], rows).reshape(B, GROUP - 9, DIM_P)
    P2 = P1.clone()
    cols = P1[:, :, 9:GROUP].reshape(B, DIM_P, N_SLOTS, 3)
    P2[:, :, 9:GROUP] = torch.where(
        mr.permute(0, 2, 1, 3), P1[:, :, 6:9][:, :, None], cols
    ).reshape(B, DIM_P, GROUP - 9)
    add = _slot_blockdiag(N_blk[:, None] * m[:, :, None].to(dtype))
    P2[:, 9:GROUP, 9:GROUP] = P2[:, 9:GROUP, 9:GROUP] + add
    return State(X_new, state.theta, _sym(P2),
                 state.jitter_count, state.info_count)


def step(state: State, imu, p_meas, dt_row, prop_mask, correct_mask,
         insert_mask, covs, R_kin, s_jitter: float) -> tuple[State, RowOut]:
    """One filter row: propagate -> masked correct -> masked insert.

    Removal needs no operation (mask clear happens in the schedule). Padded
    rows (dt_row = 0, all-false masks) are exact no-ops.
    """
    state = _propagate(state, imu[:, 0:3], imu[:, 3:6], dt_row, prop_mask, covs)
    R_pre = state.X[:, 0:3, 0:3]
    state, nis, N_blk = _correct(state, p_meas, correct_mask, R_kin, s_jitter)
    state = _insert(state, p_meas, insert_mask, R_pre, N_blk)
    out = RowOut(state.X[:, 0:3, 0:3], state.X[:, 0:3, 3], state.X[:, 0:3, 4],
                 nis)
    return state, out


def apply_row0(state: State, p_meas0, insert0, R_kin) -> State:
    """Segment-start row: pure insertion (mirrors ``start_filter``: no slots
    exist yet, so the row-0 kinematic pass augments and never corrects)."""
    R_pre = state.X[:, 0:3, 0:3]
    N_blk = R_pre @ R_kin @ R_pre.transpose(1, 2)
    return _insert(state, p_meas0, insert0, R_pre, N_blk)


def run_rows_fixed(state: State, batch: BatchData, rows: slice, covs,
                   R_kin, *, s_jitter: float = 0.0,
                   step_fn: Callable | None = None) -> tuple[State, dict]:
    """Advance over batch rows [rows.start, rows.stop) (local indices >= 1).

    Returns (state, out) with out tensors shaped (B, T_chunk, ...). The caller
    detaches state at truncated-BPTT chunk boundaries via ``detach_state``.
    """
    fn = step if step_fn is None else step_fn
    R_out, v_out, p_out, nis_out = [], [], [], []
    for t in range(rows.start, rows.stop):
        state, out = fn(state, batch.imu[:, t], batch.p_meas[:, t],
                        batch.dt_row[:, t], batch.prop_mask[:, t],
                        batch.correct_mask[:, t], batch.insert_mask[:, t],
                        covs, R_kin, s_jitter)
        R_out.append(out.R)
        v_out.append(out.v)
        p_out.append(out.p)
        nis_out.append(out.nis)
    return state, {
        "R_WB": torch.stack(R_out, dim=1),
        "v_W": torch.stack(v_out, dim=1),
        "p_W": torch.stack(p_out, dim=1),
        "nis": torch.stack(nis_out, dim=1),
        "nis_dim": batch.nis_dim[:, rows],
    }


def reg_nis_masked(nis: torch.Tensor, nis_dim: torch.Tensor) -> torch.Tensor:
    """(mean(NIS/dim) - 1)^2 over rows that had measurements; exactly the
    masked equivalent of covariance_calibration.reg_nis."""
    has = nis_dim > 0
    per_dim = torch.where(has, nis / nis_dim.clamp_min(1.0),
                          torch.zeros_like(nis))
    count = has.sum().clamp_min(1)
    return (per_dim.sum() / count - 1.0) ** 2


def make_compiled_step(mode: str | None) -> Callable:
    """step compiled with fullgraph=True, or eager for mode None."""
    if mode is None:
        return step
    return torch.compile(step, fullgraph=True, mode=mode)
