# Python Paper Implementation

Python implementation of the bilevel estimation-calibration pipeline, using the B1 data and URDF resources in this repository. The lower level is a CasADi/Fatrop full-information estimator (FIE); the upper level computes gradients through the estimator KKT system and updates the parameters — FIE weights, foot-tip offsets, and base/mocap alignment — with Frank-Wolfe steps and an Armijo line search.

## Code Path

All code lives in `bilevel/`.

- Configuration and data: `config.py` holds runtime defaults, resource paths, and the index layout of `theta = [FIE weights | tip offsets | base offset]`; `data_io.py` loads the B1 CSVs and applies downsampling, foot z offsets, contact thresholding, and window slicing.
- Robot and kinematics: `robot.py` wraps the Pinocchio B1 model and builds measurement vectors, measurement Jacobians, and the initial prior, using helpers from `kinematics.py` and the generated CasADi functions loaded by `codegen.py` from `resources/codegen`.
- Lower-level estimator: `estimator/full_information.py` builds the FIE NLP (dynamics constraints, measurement and process costs) and solves it with Fatrop; it also builds the KKT derivative functions and caches them on disk.
- Upper-level gradient: `sensitivity.py` assembles the KKT Jacobians at the solution and solves the sparse system for `dstate/dtheta`, including the measurement dependence on the tip offsets.
- Calibration loop: `calibration.py` runs the Frank-Wolfe iterations with the trajectory loss from `losses.py`, the CVXPY feasible-set LMO from `lmo.py`, and an Armijo line search.
- Entry point and exports: `run_bilevel.py` is the CLI; `exports.py` writes the CSV snapshots and plots.

## Resources

- `resources/data/b1/` — B1 mocap CSV dataset
- `resources/robot/B1.urdf` — robot model
- `resources/codegen/` — generated CasADi kinematics libraries
- `resources/poster/poster.pdf`
- Slides: https://slides.com/denglincheng/icra26

## Environment

Install from the repository root:

```bash
conda create -n estimation_calibration python=3.10
conda activate estimation_calibration
conda install -c conda-forge pinocchio casadi numpy scipy matplotlib
pip install -e .
```

CasADi must provide the Fatrop NLP plugin (`casadi.has_nlpsol("fatrop")` is checked at startup). MOSEK is the default LMO solver (`FrankWolfeConfig.lmo_solver`). If using pip for Pinocchio instead of conda, install with `pip install -e ".[robotics]"`.

## Run

```bash
PYTHONPATH=src/python python3 -m bilevel.run_bilevel
```

or, after `pip install -e .`:

```bash
estimation-calibration
```

## Defaults

From `BilevelConfig` (paths relative to the repository root):

- data: `src/python/resources/data/b1/`
- robot: `src/python/resources/robot/B1.urdf`
- generated libraries: `src/python/resources/codegen/`
- CasADi derivative cache: `.cache/casadi/B1_H3000/`
- outputs: `outputs/`
- default window starts at index 22000 with horizon 3000

## Outputs

Each run writes start/end and per-iteration trajectory snapshots (CSV), theta history, position/velocity/quaternion/feet comparison plots, and tip/base offset history plots to the output directory. Built KKT derivative functions are cached under `.cache/casadi/` and reused across runs.

## Troubleshooting

- Missing Fatrop plugin: install a CasADi build with `nlpsol("fatrop")`.
- Missing Pinocchio: install via conda-forge or `pip install -e ".[robotics]"`.
- Missing MOSEK or license: change `FrankWolfeConfig.lmo_solver` to another CVXPY-supported solver.
- Generated-code libraries: the loader looks for `.so` / `.dylib` / `.dll` under `resources/codegen`; the repository ships Linux `.so` binaries, so other platforms need the libraries rebuilt from the shipped C sources.
