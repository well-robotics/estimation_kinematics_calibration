# Estimation Calibration

Code for **Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization**.

Paper: https://arxiv.org/pdf/2510.11539

The paper calibrates process covariance, measurement covariance, foot-tip offsets, and base/mocap alignment by putting a full-information estimator inside a Frank-Wolfe outer loop. The lower-level estimator is formulated with CasADi/Fatrop, and upper-level gradients are computed from the estimator KKT system.

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── cuda/                 # installable Torch CPU/CUDA calibration package
└── src/
    ├── matlab/           # planar MATLAB reference implementation
    └── python/
        ├── bilevel/      # paper-aligned Python implementation
        └── resources/    # B1 data, URDF, codegen libraries, poster
```

## Implementations

- `src/python` — paper-aligned Python implementation: CasADi/Fatrop full-information estimator, KKT sensitivities, Frank-Wolfe updates. See [`src/python/README.md`](src/python/README.md).
- `src/matlab` — compact planar 2-D reference implementation of the same estimator-in-the-loop calibration. See [`src/matlab/README.md`](src/matlab/README.md).
- `cuda` — split-safe Torch covariance calibration with CPU examples and CUDA acceleration. See [`cuda/README.md`](cuda/README.md).
