"""Concise package and user-surface checks."""

from __future__ import annotations

import json
from pathlib import Path


CUDA_ROOT = Path(__file__).parents[1]


def test_package_metadata_and_license_agree():
    metadata = (CUDA_ROOT / "pyproject.toml").read_text()
    build, project = metadata.split("[project]", maxsplit=1)
    project = project.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert 'requires = ["setuptools>=77", "wheel"]' in build
    assert 'version = "0.2.0"' in project
    assert 'license = "MIT"' in project
    assert 'license-files = ["LICENSE"]' in project
    assert '"torch>=2.11.0"' in project
    assert "matplotlib" not in project
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
