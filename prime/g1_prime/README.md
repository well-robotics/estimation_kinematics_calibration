# G1 PRIME covariance calibration

Reproducible bilevel covariance calibration for the Unitree G1. PRIME FDDP and
contact Newton solves form the lower problem; SQP--BFGS or Frank--Wolfe--SDP
minimizes the upper SE(3)-log trajectory loss.

The released upper problem varies the joint-position measurement block and
directly minimizes SE(3)-log loss on the two 501-state clips; all other
covariance coordinates remain fixed, and no accuracy beyond these clips is
claimed.

## Quickstart

Build the PRIME lower solver and install the `g1cal` control plane:

```bash
conda env create -f environment.yml
conda activate g1cal
./scripts/build.sh
python -m pip install -e .
g1cal solve --clip run1 --covariance data/calibrated/precision.csv
```

## Repository map

| Directory | Responsibility |
|---|---|
| [`configs/`](configs/README.md) | Lower-solver configuration |
| [`cpp/`](cpp/README.md) | PRIME overlay, lower-solver executable, pybind module, and contact model |
| [`data/`](data/README.md) | Released clips, calibrated covariance, and reference solutions |
| [`models/`](models/README.md) | Pinned G1 URDF, MJCF, meshes, contact frames, and manifest |
| [`python/`](python/README.md) | Installable `g1cal` calibration and solver package |
| [`scripts/`](scripts/) | Build and release-maintenance entry points |
| [`third_party/`](third_party/) | Pinned PRIME source and preserved notices |

## Calibration architecture

`CalibrationOracle` maps each optimizer coordinate to block covariance and
precision, runs both 501-state PRIME lower problems, and returns their mean
SE(3)-log loss. Whole-estimator central differences supply the SQP--BFGS or
Frank--Wolfe--SDP update.

## Commands

| Command | Purpose |
|---|---|
| `g1cal solve` | Run one lower estimator at a selected covariance |
| `g1cal calibrate` | Run SQP--BFGS or Frank--Wolfe--SDP upper updates |
| `g1cal select` | Select the lowest strict evaluated covariance |

Use `g1cal COMMAND --help` for the exact arguments.

## Reproduce the calibration

Run either method through the shared content-addressed oracle and strict
promotion gate:

```bash
g1cal calibrate --optimizer sqp-bfgs --max-iterations 2 \
  --out out/calibration
g1cal calibrate --optimizer frank-wolfe-sdp --max-iterations 2 \
  --out out/calibration
g1cal select --out out/calibration
```

Lower attempts are immutable; whole-estimator central differences drive both
methods, and the Frank--Wolfe SDP oracle is checked against the analytic
interval endpoint.

## Acknowledgments and licenses

Built on the excellent work of
[PRIME](https://github.com/well-robotics/PRIME) (well-robotics), BSD-3. PRIME
provides the lower estimator's contact machinery; its license and notices are
preserved under [`third_party/PRIME/`](third_party/PRIME/VENDORED.md).

Unitree G1 model provenance and license text are recorded in
[`models/g1/NOTICE.md`](models/g1/NOTICE.md). Repository code in this subtree is
BSD-3-Clause; third-party material remains subject to its notices.

Return to the [PRIME implementations](../README.md).
