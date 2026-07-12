"""Shared fixtures: real-rollout data, covariance modules, both filter impls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from estimation_calibration_cuda.covariance_calibration import (
    CalibrationConfig,
    build_covs,
    fixed_initial_covariance,
    load_rollout,
    make_cov_modules,
    seed_state,
)
from estimation_calibration_cuda.invariant_ekf import run_rows, start_filter
from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.data_paths import (
    leg_bical_data_root,
)

DATA_ROOT = leg_bical_data_root()
SYNTHETIC_GOLDEN = (
    Path(__file__).resolve().parent / "data" / "legacy_synthetic_mini_golden.npz"
)
STEM = "dance1_subject1_20260623_173019"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def config():
    return CalibrationConfig()


@pytest.fixture(scope="session")
def real_data_root():
    if DATA_ROOT is None or not DATA_ROOT.is_dir():
        pytest.skip("external data not configured")
    manifest = DATA_ROOT / "dataset_manifest.json"
    if not manifest.is_file():
        pytest.fail("configured data root has no dataset_manifest.json")
    return DATA_ROOT


@pytest.fixture(scope="session")
def roll(device, config, real_data_root):
    required = [
        real_data_root / f"{STEM}.npz",
        real_data_root / f"{STEM}.features.npz",
    ]
    if any(not path.is_file() for path in required):
        pytest.fail(f"configured data root has no complete pair for {STEM}")
    return load_rollout(real_data_root, STEM, "test", config=config,
                        device=device)


@pytest.fixture(scope="session")
def real_rollouts(device, config, real_data_root):
    manifest = json.loads((real_data_root / "dataset_manifest.json").read_text())
    stems = sorted({Path(entry["dataset_path"]).stem for entry in manifest})
    incomplete = [
        stem for stem in stems
        if not (real_data_root / f"{stem}.npz").is_file()
        or not (real_data_root / f"{stem}.features.npz").is_file()
    ]
    if incomplete:
        pytest.fail("configured data root has incomplete rollout pairs")
    if not stems:
        pytest.fail("configured data manifest has no rollouts")
    return [
        load_rollout(real_data_root, stem, "test", config=config,
                     device=device)
        for stem in stems
    ]


@pytest.fixture(scope="session")
def synthetic_golden():
    with np.load(SYNTHETIC_GOLDEN, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@pytest.fixture(scope="session")
def covs_pair(device):
    torch.manual_seed(0)
    modules = make_cov_modules(device=device)
    covs, R_kin = build_covs(modules)
    return ({k: v.detach() for k, v in covs.items()}, R_kin.detach())


@pytest.fixture(scope="session")
def P0_fixed(device):
    return fixed_initial_covariance(device)


def run_dynamic(roll, covs, R_kin, P0_fixed, config, rows=None):
    """start_filter + run_rows over the trimmed segment (training convention)."""
    s0 = roll.trim0
    s1 = roll.trim1 if rows is None else min(s0 + 1 + rows, roll.trim1)
    X0, theta0, P0 = seed_state(roll, s0, P0_fixed)
    filt = start_filter(X0, theta0, P0, covs, roll.flags[s0], roll.p_BC[s0],
                        R_kin, s_jitter=config.s_jitter)
    out = run_rows(filt, roll.imu[s0 + 1:s1], roll.dt, roll.p_BC[s0 + 1:s1],
                   None, None, R_kin, collect_nis=True,
                   changes_list=roll.changes[s0 + 1:s1])
    return filt, out


def run_fixed(roll, covs, R_kin, P0_fixed, config, rows=None, step_fn=None):
    s0 = roll.trim0
    n = (roll.trim1 - roll.trim0 - 1) if rows is None \
        else min(rows, roll.trim1 - roll.trim0 - 1)
    batch = fsi.build_batch([roll])
    state = fsi.init_state([seed_state(roll, s0, P0_fixed)],
                           device=roll.imu.device)
    state = fsi.apply_row0(state, batch.p_meas[:, 0], batch.insert_mask[:, 0],
                           R_kin)
    out = None
    if n > 0:
        state, out = fsi.run_rows_fixed(state, batch, slice(1, 1 + n), covs,
                                        R_kin, s_jitter=config.s_jitter,
                                        step_fn=step_fn)
    return state, out, batch


def dynamic_column_map(flags_seg: np.ndarray) -> dict[int, int]:
    """Simulate the dynamic filter's contact-column bookkeeping over a flags
    segment (row 0 = segment start): returns final {candidate_id: X column}."""
    flags = np.asarray(flags_seg).astype(bool)
    pos: dict[int, int] = {}
    dim_x = 5
    prev = np.zeros(flags.shape[1], dtype=bool)
    for k in range(flags.shape[0]):
        row = flags[k]
        removals = sorted((cid for cid in list(pos)
                           if prev[cid] and not row[cid]),
                          key=lambda c: pos[c], reverse=True)
        for cid in removals:
            col = pos.pop(cid)
            dim_x -= 1
            for c2 in pos:
                if pos[c2] > col:
                    pos[c2] -= 1
        for cid in range(flags.shape[1]):
            if row[cid] and cid not in pos:
                pos[cid] = dim_x
                dim_x += 1
        prev = row
    return pos


def map_dynamic_P_to_fixed(P_dyn: torch.Tensor, pos: dict[int, int],
                           P_fix: torch.Tensor):
    """Extract matching (active-block) submatrices from both layouts."""
    dim_p = P_dyn.shape[0]
    order = sorted(pos, key=lambda c: pos[c])
    dyn_idx = list(range(9))
    fix_idx = list(range(9))
    for cid in order:
        dyn_idx += [3 * pos[cid] - 6 + j for j in range(3)]
        fix_idx += [9 + 3 * cid + j for j in range(3)]
    dyn_idx += [dim_p - 6 + i for i in range(6)]
    fix_idx += [fsi.GROUP + i for i in range(6)]
    return P_dyn[dyn_idx][:, dyn_idx], P_fix[fix_idx][:, fix_idx]
