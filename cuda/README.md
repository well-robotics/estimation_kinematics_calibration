# Estimation Calibration CUDA

CUDA PyTorch prototype for learning covariance parameters in a contact-aided right-invariant InEKF. The filter replay is differentiable, so covariance parameters can be optimized with standard Torch training loops instead of a symbolic bilevel solver.

## Layout

```text
cuda/
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── covariance_tuning_tutorial.ipynb
│   ├── covariance_calibration_playground.ipynb
│   └── covariance_calibration_run.ipynb
└── src/estimation_calibration_cuda/
    ├── __init__.py
    ├── invariant_ekf.py
    └── covariance_calibration.py
```

## Environment

This folder is a standalone uv project.

CUDA PyTorch environment:

```bash
cd cuda
uv sync --extra cu130 --extra notebooks
```

CPU-only inspection environment:

```bash
cd cuda
uv sync --extra cpu --extra notebooks
```

Verify Torch:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

The training path expects CUDA float64. CPU mode is useful for reading notebooks, inspecting saved JSON, and lightweight imports.

## Notebooks

- `notebooks/covariance_tuning_tutorial.ipynb` is a small SO(3) toy tutorial that shows end-to-end differentiable covariance tuning with Adam and SGD.
- `notebooks/covariance_calibration_playground.ipynb` is self-contained. It builds a Torch estimator, starts with bad covariance matrices, and trains them on synthetic data.
- `notebooks/covariance_calibration_run.ipynb` is the larger calibration notebook for real rollout datasets.

## Rerun Training

Provide the raw dataset directory and run:

```bash
uv run estimation-calibration-cuda train \
  --data-root /path/to/datasets_v0 \
  --outputs runs/covariance_calibration
```

Expected dataset files:

```text
datasets_v0/
├── dataset_manifest.json
├── <rollout>.npz
└── <rollout>.features.npz
```

Training writes calibrated covariances, a checkpoint, plots, and JSON summaries under the selected output directory.
