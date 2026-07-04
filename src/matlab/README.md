# MATLAB Reference Implementation

Planar 2-D full-information estimation with Frank-Wolfe covariance calibration. The lower-level estimator is solved with CasADi/IPOPT; the upper level uses finite-difference sensitivities and a YALMIP-based linear minimization oracle. This is a compact reference implementation, separate from the B1 Python pipeline in `src/python`.

## Code Path

- `FIE.m` — 8-state planar estimator (`[base_position; base_velocity; left_foot; right_foot]`). Builds arrival, dynamics, and measurement terms from the `q`/`dq`/`ddq`/contact histories and solves the full-information problem with CasADi/IPOPT; provides finite-difference Jacobians with optional parallel evaluation.
- `FIECalibrator.m` — Frank-Wolfe loop: loss against ground truth, finite-difference gradients, YALMIP LMO with PSD constraints on the covariance blocks, history and trajectory export.
- `calibrationOptions.m` — default `theta`, bounds, solver options, output settings.
- `runCalibration.m` — main entry point; accepts MAT files or structs and normalizes the old log format and newer data structs.
- `plotFIE.m`, `main.m`, `estimation_FIE.m` — plotting and compatibility wrappers.

## Usage

```matlab
options = calibrationOptions();
result = runCalibration(inputData, options);
```

`inputData` may be a MAT file path or a struct with either:

- `log.flow.q`, `log.flow.dq`, `log.flow.ddq`, `log.estimate.contact`, `log.estimate.t`, `log.groundtruth.x`
- `data.q`, `data.dq`, `data.ddq`, `data.contact`, `data.dt`, plus `xGroundTruth` or `xGT`

State trajectories are `8-by-K`; `q`, `dq`, `ddq` are `7-by-K`. Contact uses `-1` for left foot, `0` for double support, `1` for right foot.

## Kinematics

The default model expects `pLeftToe_d`, `pRightToe_d`, `J_leftToe_d`, and `J_rightToe_d` on the MATLAB path. Equivalent function handles can instead be passed through the `model` argument of `FIE`.

## Outputs

Written to the output directory from `calibrationOptions.m` (default `outputs/matlab` under the current directory): `calibration_history.csv`, `theta_history.csv`, and `trajectory_ground_truth_final.csv`.

## Dependencies

MATLAB, CasADi with IPOPT, YALMIP, and MOSEK or another YALMIP-compatible SDP solver. Add external dependencies to the MATLAB path before running.
