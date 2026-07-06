"""Gradient checks: finite/nonzero, dynamic-vs-fixed parity, compiled-vs-eager."""

from __future__ import annotations

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.covariance_calibration import (
    GROUP_ORDER,
    build_covs,
    covariance_regularization,
    make_cov_modules,
    seed_state,
)
from estimation_calibration_cuda.invariant_ekf import run_rows, start_filter

ROWS = 300


def _dynamic_loss(roll, modules, P0_fixed, config):
    covs, R_kin = build_covs(modules)
    s0 = roll.trim0
    X0, theta0, P0 = seed_state(roll, s0, P0_fixed)
    filt = start_filter(X0, theta0, P0, covs, roll.flags[s0], roll.p_BC[s0],
                        R_kin, s_jitter=config.s_jitter)
    out = run_rows(filt, roll.imu[s0 + 1:s0 + 1 + ROWS], roll.dt,
                   roll.p_BC[s0 + 1:s0 + 1 + ROWS], None, None, R_kin,
                   collect_nis=True,
                   changes_list=roll.changes[s0 + 1:s0 + 1 + ROWS])
    v_B = torch.einsum("tji,tj->ti", out["R_WB"], out["v_W"])
    loss_body = ((v_B - roll.gt_v_B[s0 + 1:s0 + 1 + ROWS]) ** 2).sum(-1).mean()
    loss_reg, _ = covariance_regularization(
        modules, out["nis_values"], out["nis_dims"], device=loss_body.device)
    return loss_body + loss_reg


def _fixed_loss(roll, modules, P0_fixed, config, step_fn=None):
    covs, R_kin = build_covs(modules)
    batch = fsi.build_batch([roll])
    state = fsi.init_state([seed_state(roll, roll.trim0, P0_fixed)],
                           device=roll.imu.device)
    state = fsi.apply_row0(state, batch.p_meas[:, 0], batch.insert_mask[:, 0],
                           R_kin)
    state, out = fsi.run_rows_fixed(state, batch, slice(1, 1 + ROWS), covs,
                                    R_kin, s_jitter=config.s_jitter,
                                    step_fn=step_fn)
    v_B = torch.einsum("btji,btj->bti", out["R_WB"], out["v_W"])
    gt = roll.gt_v_B[roll.trim0 + 1:roll.trim0 + 1 + ROWS][None]
    loss_body = ((v_B - gt) ** 2).sum(-1).mean()
    # same regularization; NIS term computed from the masked stream
    loss_reg, terms = covariance_regularization(
        modules, [], [], device=loss_body.device)
    nis_vals = out["nis"][out["nis_dim"] > 0]
    nis_dims = out["nis_dim"][out["nis_dim"] > 0]
    from estimation_calibration_cuda.covariance_calibration import LAMBDA
    nis_term = LAMBDA["nis"] * ((nis_vals / nis_dims).mean() - 1.0) ** 2
    return loss_body + loss_reg + nis_term


def _grads(modules, loss):
    for p in modules.parameters():
        p.grad = None
    loss.backward()
    return {name: modules[name].raw_tril.grad.detach().clone()
            for name in GROUP_ORDER}


def _fresh_modules(device):
    torch.manual_seed(0)
    return make_cov_modules(device=device)


def test_grads_finite_nonzero_and_match_dynamic(roll, P0_fixed, config, device):
    mod_d = _fresh_modules(device)
    grads_d = _grads(mod_d, _dynamic_loss(roll, mod_d, P0_fixed, config))
    mod_f = _fresh_modules(device)
    grads_f = _grads(mod_f, _fixed_loss(roll, mod_f, P0_fixed, config))
    nonzero = 0
    for name in GROUP_ORDER:
        gd, gf = grads_d[name], grads_f[name]
        assert torch.isfinite(gd).all() and torch.isfinite(gf).all(), name
        nonzero += int(gd.abs().max() > 0)
        assert torch.allclose(gd, gf, rtol=1e-6, atol=1e-12), (
            name, float((gd - gf).abs().max()))
    assert nonzero >= 4  # most groups receive signal on a real chunk


def test_compiled_grads_match_eager(roll, P0_fixed, config, device):
    step_c = fsi.make_compiled_step("default")
    mod_e = _fresh_modules(device)
    grads_e = _grads(mod_e, _fixed_loss(roll, mod_e, P0_fixed, config))
    mod_c = _fresh_modules(device)
    grads_c = _grads(mod_c, _fixed_loss(roll, mod_c, P0_fixed, config,
                                        step_fn=step_c))
    for name in GROUP_ORDER:
        d = float((grads_e[name] - grads_c[name]).abs().max())
        assert d <= 1e-8, (name, d)
        assert torch.isfinite(grads_c[name]).all()
