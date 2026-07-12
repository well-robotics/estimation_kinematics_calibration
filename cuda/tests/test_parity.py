"""Replay parity: fixed-slot vs dynamic oracle, and vs the Gate C golden."""

from __future__ import annotations

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi

from conftest import (
    dynamic_column_map,
    map_dynamic_P_to_fixed,
    run_dynamic,
    run_fixed,
)

pytestmark = pytest.mark.external_data


def _traj_maxdiff(out_d, out_f):
    return {
        "R": float((out_d["R_WB"] - out_f["R_WB"][0]).abs().max()),
        "v": float((out_d["v_W"] - out_f["v_W"][0]).abs().max()),
        "p": float((out_d["p_W"] - out_f["p_W"][0]).abs().max()),
    }


@pytest.mark.parametrize("rows,tol", [(2000, 1e-9)])
def test_slice_parity(roll, covs_pair, P0_fixed, config, rows, tol):
    covs, R_kin = covs_pair
    with torch.no_grad():
        filt, out_d = run_dynamic(roll, covs, R_kin, P0_fixed, config, rows)
        state, out_f, _ = run_fixed(roll, covs, R_kin, P0_fixed, config, rows)
    diffs = _traj_maxdiff(out_d, out_f)
    assert max(diffs.values()) <= tol, diffs
    nis_d = torch.stack(out_d["nis_values"])
    nis_f = out_f["nis"][0][out_f["nis_dim"][0] > 0]
    assert nis_d.numel() == nis_f.numel()
    assert float((nis_d - nis_f).abs().max()) <= tol
    pos = filt.estimated_contact_positions
    Pd, Pf = map_dynamic_P_to_fixed(filt.P, pos, state.P[0])
    assert float((Pd - Pf).abs().max()) <= tol
    assert filt.jitter_events == int(state.jitter_count.item())


def test_full_rollout_parity(roll, covs_pair, P0_fixed, config):
    covs, R_kin = covs_pair
    with torch.no_grad():
        filt, out_d = run_dynamic(roll, covs, R_kin, P0_fixed, config)
        state, out_f, batch = run_fixed(roll, covs, R_kin, P0_fixed, config)
    diffs = _traj_maxdiff(out_d, out_f)
    print(f"full-rollout drift ({roll.trim1 - roll.trim0 - 1} rows): {diffs}")
    assert max(diffs.values()) <= 1e-7, diffs
    # robust gate: the training metric agrees to numerical noise
    gt = roll.gt_v_B[roll.trim0 + 1:roll.trim1]
    def vb_rmse(R, v):
        v_B = torch.einsum("tji,tj->ti", R, v)
        return float(torch.sqrt(((v_B - gt) ** 2).sum(-1).mean()))
    rmse_d = vb_rmse(out_d["R_WB"], out_d["v_W"])
    rmse_f = vb_rmse(out_f["R_WB"][0], out_f["v_W"][0])
    assert abs(rmse_d - rmse_f) <= 1e-9
    # final covariance parity on active blocks, and PSD/symmetry
    Pd, Pf = map_dynamic_P_to_fixed(
        filt.P, filt.estimated_contact_positions, state.P[0])
    assert float((Pd - Pf).abs().max()) <= 1e-7
    Pf_full = state.P[0]
    assert torch.isfinite(Pf_full).all()
    assert float((Pf_full - Pf_full.T).abs().max()) < 1e-9
    assert float(torch.linalg.eigvalsh(Pd).min()) > -1e-12


def test_column_map_matches_dynamic(roll, covs_pair, P0_fixed, config):
    """The host-side bookkeeping simulation reproduces the filter's columns."""
    covs, R_kin = covs_pair
    with torch.no_grad():
        filt, _ = run_dynamic(roll, covs, R_kin, P0_fixed, config, rows=3000)
    sim = dynamic_column_map(roll.flags[roll.trim0:roll.trim0 + 1 + 3000])
    assert sim == filt.estimated_contact_positions


def test_padded_rows_are_bitwise_noop(roll, covs_pair, P0_fixed, config):
    covs, R_kin = covs_pair
    with torch.no_grad():
        state, _, batch = run_fixed(roll, covs, R_kin, P0_fixed, config,
                                    rows=500)
        B = batch.B
        dev = roll.imu.device
        zeros8 = torch.zeros(B, fsi.N_SLOTS, dtype=torch.bool, device=dev)
        state2, out = fsi.step(
            state, torch.zeros(B, 6, dtype=torch.float64, device=dev),
            torch.zeros(B, fsi.N_SLOTS, 3, dtype=torch.float64, device=dev),
            torch.zeros(B, dtype=torch.float64, device=dev),
            zeros8, zeros8, zeros8, covs, R_kin, config.s_jitter)
    assert torch.equal(state2.X, state.X)
    assert torch.equal(state2.theta, state.theta)
    assert torch.equal(state2.P, state.P)
    assert torch.equal(state2.jitter_count, state.jitter_count)
    assert float(out.nis.abs().max()) == 0.0


def test_chunked_equals_monolithic(roll, covs_pair, P0_fixed, config):
    covs, R_kin = covs_pair
    n = 900
    with torch.no_grad():
        _, out_mono, _ = run_fixed(roll, covs, R_kin, P0_fixed, config, rows=n)
        state, _, batch = run_fixed(roll, covs, R_kin, P0_fixed, config, rows=0)
        outs = []
        for a in range(1, 1 + n, 300):
            state, o = fsi.run_rows_fixed(state, batch, slice(a, a + 300),
                                          covs, R_kin,
                                          s_jitter=config.s_jitter)
            state = fsi.detach_state(state)
            outs.append(o)
    for key in ["R_WB", "v_W", "p_W", "nis"]:
        cat = torch.cat([o[key] for o in outs], dim=1)
        assert torch.equal(cat, out_mono[key]), key


def test_batched_equals_single(device, config, P0_fixed, covs_pair,
                               real_rollouts):
    """B=7 padded batch reproduces per-rollout runs (bmm vs mm tolerance)."""
    from estimation_calibration_cuda.covariance_calibration import seed_state
    covs, R_kin = covs_pair
    rolls = real_rollouts
    n = 1500
    with torch.no_grad():
        batch = fsi.build_batch(rolls)
        state = fsi.init_state(
            [seed_state(r, r.trim0, P0_fixed) for r in rolls], device=device)
        state = fsi.apply_row0(state, batch.p_meas[:, 0],
                               batch.insert_mask[:, 0], R_kin)
        state, out_b = fsi.run_rows_fixed(state, batch, slice(1, 1 + n),
                                          covs, R_kin,
                                          s_jitter=config.s_jitter)
        for i, r in enumerate(rolls):
            _, out_1, _ = run_fixed(r, covs, R_kin, P0_fixed, config, rows=n)
            for key in ["R_WB", "v_W", "p_W", "nis"]:
                # bmm vs mm reduction-order differences random-walk over rows;
                # scale-aware bound (position is unbounded on running rollouts)
                ref = out_1[key][0]
                d = float((out_b[key][i] - ref).abs().max())
                tol = 1e-8 * max(1.0, float(ref.abs().max()))
                assert d <= tol, (r.stem, key, d, tol)
