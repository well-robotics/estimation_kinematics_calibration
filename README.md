# Estimation Calibration

Code for **Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization**.

Paper: https://arxiv.org/pdf/2510.11539

The paper version calibrates process covariance, measurement covariance, foot-tip offsets, and base/mocap alignment by putting a full-information estimator inside a Frank-Wolfe outer loop. The lower-level estimator is formulated with CasADi/Fatrop, and upper-level gradients are computed from the estimator KKT system.

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── cuda/                 # CUDA/PyTorch update
└── src/
    ├── matlab/           # MATLAB reference implementation
    └── python/
        ├── bilevel/      # paper-style Python implementation
        └── resources/    # B1 data, URDF, codegen libraries, poster
```

`src/python` and `src/matlab` are the implementations aligned with the paper. `cuda/` is a update: it moves the estimator replay and covariance tuning into CUDA PyTorch tensors. This makes it easier to train on more rollout data and compute gradients with Torch autograd instead of symbolic KKT differentiation.

## CUDA/PyTorch Update

The `cuda/` folder is not part of the paper implementation. It is a update for running the estimator and covariance calibration directly on CUDA PyTorch tensors:

- differentiable filter replay in Torch by BPTT
- covariance training with Adam
- easier batching over more datasets/rollouts
- faster gradient computation on GPU

See `cuda/README.md` for uv setup and notebooks.

Quick start:

```bash
cd cuda
uv sync --extra cu130 --extra notebooks
```

## Python Paper Version

Install from the repository root:

```bash
conda create -n estimation_calibration python=3.10
conda activate estimation_calibration
conda install -c conda-forge pinocchio casadi numpy scipy matplotlib
pip install -e .
```

Notes:

- CasADi must provide the `fatrop` NLP plugin. The entry point checks `casadi.has_nlpsol("fatrop")` before running.
- The default LMO solver is MOSEK. Install and license MOSEK for full runs, or change `FrankWolfeConfig.lmo_solver` in `src/python/bilevel/config.py`.
- If using pip for Pinocchio instead of conda, install with `pip install -e ".[robotics]"`.

Run:

```bash
PYTHONPATH=src/python python3 -m bilevel.run_bilevel
```

If installed with `pip install -e .`, the script entry point is also available:

```bash
estimation-calibration
```

Default resources:

- data: `src/python/resources/data/b1/`
- robot model: `src/python/resources/robot/B1.urdf`
- generated CasADi libraries: `src/python/resources/codegen/`
- poster: `src/python/resources/poster/poster.pdf`
- slides: https://slides.com/denglincheng/icra26

Generated files are ignored by git:

```text
.cache/casadi/B1_H3000/
outputs/
```