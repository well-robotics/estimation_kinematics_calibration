"""Concise package and user-surface checks."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path


CUDA_ROOT = Path(__file__).parents[1]


def test_package_metadata_and_license_agree():
    metadata = tomllib.loads((CUDA_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["version"] == "0.2.0"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert "torch>=2.11.0" in project["dependencies"]
    assert not any("matplotlib" in item for item in project["dependencies"])
    assert metadata["build-system"]["requires"][0] == "setuptools>=77"
    assert (CUDA_ROOT / "LICENSE").read_text().startswith("MIT License\n")


def test_readme_and_notebook_are_compact_and_clean():
    readme = (CUDA_ROOT / "README.md").read_text()
    assert 100 <= len(readme.splitlines()) <= 140
    assert "estimation-calibration-cuda train example" in readme
    assert "contact_process_covariance" in readme
    notebooks = list((CUDA_ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["covariance_tuning_tutorial.ipynb"]
    notebook = json.loads(notebooks[0].read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
