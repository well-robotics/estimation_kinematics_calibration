"""Public API, resume, CLI, and artifact contracts."""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil

import numpy as np
import pytest
import torch

import estimation_calibration_cuda as public
from estimation_calibration_cuda import (
    CalibrationConfig,
    calibrate,
    evaluate,
    load_dataset,
)
from estimation_calibration_cuda.api import inspect_run


def _config(epochs=1, **changes):
    values = dict(device="cpu", compile_mode="none", epochs=epochs,
                  chunk=32, seed=31)
    values.update(changes)
    return CalibrationConfig(**values)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "run"
    calibrate(load_dataset("example"), _config(), output_dir=path)
    return path


def test_public_surface_is_exact():
    assert public.__all__ == [
        "CalibrationConfig", "CalibrationEpisode", "CalibrationResult",
        "load_dataset", "calibrate", "evaluate",
    ]


def test_calibrate_is_split_lazy_and_writes_four_files(tmp_path, monkeypatch):
    calls = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        calls.append(getattr(path, "name", str(path)))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    before = torch.get_default_dtype()
    result = calibrate(load_dataset("example"), _config(), output_dir=tmp_path / "run")
    assert {path.name for path in result.run_dir.iterdir()} == {
        "checkpoint.pt", "covariances.npz", "metrics.json", "manifest.json"}
    assert calls[:2] == ["train.npz", "validation.npz"]
    assert "test.npz" not in calls
    assert torch.get_default_dtype() == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.selected_epoch = 99
    assert inspect_run(result.run_dir)["test_evaluated"] is False


def test_evaluate_writes_test_once_after_run_validation(base_run, tmp_path, monkeypatch):
    run = tmp_path / "run"
    shutil.copytree(base_run, run)
    dataset = load_dataset("example")
    result = evaluate(dataset, checkpoint=run / "checkpoint.pt",
                      split="test", device="cpu")
    assert result["aggregate"]["finite"]
    assert inspect_run(run)["test_evaluated"] is True

    monkeypatch.setattr(dataset, "load", lambda split: pytest.fail("test was reopened"))
    with pytest.raises(ValueError, match="already"):
        evaluate(dataset, checkpoint=run / "checkpoint.pt",
                 split="test", device="cpu")


def test_resume_matches_uninterrupted_training(tmp_path):
    dataset = load_dataset("example")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    calibrate(dataset, _config(epochs=3), output_dir=full)
    calibrate(dataset, _config(epochs=1), output_dir=resumed)
    calibrate(dataset, _config(epochs=3), output_dir=resumed, resume=True)
    one = torch.load(full / "checkpoint.pt", weights_only=False)
    two = torch.load(resumed / "checkpoint.pt", weights_only=False)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(
                equal(a, b) for a, b in zip(left, right))
        return left == right

    for key in ("current_state_dict", "optimizer_state_dict", "best",
                "validation_curve"):
        assert equal(one[key], two[key]), key


def test_resume_rejects_identity_and_target_mutations(base_run, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(base_run, run)
    dataset = load_dataset("example")
    with pytest.raises(ValueError, match="epochs"):
        calibrate(dataset, _config(epochs=1), output_dir=run, resume=True)
    with pytest.raises(ValueError, match="configuration identity"):
        calibrate(dataset, _config(epochs=2, lr=2e-2),
                  output_dir=run, resume=True)

    altered = load_dataset("example")
    record = dataclasses.replace(altered._records[0], source_id="different-source")
    altered._records = (record, *altered._records[1:])
    with pytest.raises(ValueError, match="dataset identity"):
        calibrate(altered, _config(epochs=2), output_dir=run, resume=True)


def test_mutation_and_mixed_generation_never_inspect(base_run, tmp_path, monkeypatch):
    damaged = tmp_path / "damaged"
    shutil.copytree(base_run, damaged)
    with (damaged / "metrics.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_run(damaged)

    mixed = tmp_path / "mixed"
    shutil.copytree(base_run, mixed)
    import estimation_calibration_cuda.api as api
    monkeypatch.setattr(api, "_write_npz",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        calibrate(load_dataset("example"), _config(epochs=2),
                  output_dir=mixed, resume=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_run(mixed)


def test_invalid_execution_fails_before_episode_load(monkeypatch, tmp_path):
    dataset = load_dataset("example")
    monkeypatch.setattr(dataset, "load", lambda split: pytest.fail("arrays opened"))
    with pytest.raises(ValueError, match="requires a CUDA"):
        calibrate(dataset, _config(compile_mode="cuda-graph"),
                  output_dir=tmp_path / "run")


@pytest.mark.release_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_failure_records_stable_fallback(monkeypatch, tmp_path):
    import estimation_calibration_cuda.batched_calibration as batched

    class BrokenGraph:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("host-specific details must not be serialized")

    monkeypatch.setattr(batched, "ChunkGraph", BrokenGraph)
    run = tmp_path / "run"
    calibrate(
        load_dataset("example"),
        _config(device="cuda", compile_mode="cuda-graph"),
        output_dir=run,
    )
    execution = inspect_run(run)["execution"]
    assert execution["requested_compile_mode"] == "cuda-graph"
    assert execution["effective_compile_mode"] == "none"
    assert execution["fallback_reason_code"] == "cuda_graph_capture_failed"
    assert "host-specific" not in (run / "manifest.json").read_text()


def test_nonempty_output_requires_resume(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "keep").write_text("user data")
    with pytest.raises(FileExistsError, match="not empty"):
        calibrate(load_dataset("example"), _config(), output_dir=run)


def test_cli_help(capsys):
    from estimation_calibration_cuda.cli import main
    for command in (None, "train", "evaluate", "inspect"):
        argv = ["--help"] if command is None else [command, "--help"]
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 0
    assert "train" in capsys.readouterr().out
