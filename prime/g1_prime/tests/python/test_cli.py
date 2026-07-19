"""CLI calibrated precision input."""

from __future__ import annotations

import numpy as np

from g1cal.calibration import calibrated_theta
from g1cal.cli import _resolve_theta


def test_precision_path_resolves_exact_calibrated_theta():
    resolved = _resolve_theta("data/calibrated/precision.csv")
    assert np.array_equal(resolved, calibrated_theta())
