"""Strict dataset-boundary checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.data import (
    CalibrationEpisode,
    _episode_to_rollout,
    load_dataset,
)


def _arrays(T: int = 6, N: int = 3) -> dict[str, np.ndarray]:
    time_s = np.arange(T, dtype=np.float64) * 0.01
    return {
        "time_s": time_s,
        "imu": np.zeros((T, 6), dtype=np.float32),
        "p_BC": np.zeros((T, N, 3), dtype=np.float64),
        "contact_flags": np.ones((T, N), dtype=bool),
        "gt_R_WB": np.broadcast_to(np.eye(3), (T, 3, 3)).copy(),
        "gt_v_W": np.zeros((T, 3), dtype=np.float64),
        "gt_p_W": np.zeros((T, 3), dtype=np.float64),
    }


def _episode(**changes) -> CalibrationEpisode:
    values = _arrays()
    values.update(changes)
    return CalibrationEpisode(
        name=values.pop("name", "episode-01"),
        split=values.pop("split", "train"),
        source_id=values.pop("source_id", "source-01"),
        **values,
    )


def _write_npz(root: Path, name: str, arrays=None) -> str:
    path = root / name
    np.savez(path, **(_arrays() if arrays is None else arrays))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(name: str, split: str, filename: str, digest: str, source=None) -> dict:
    return {
        "name": name,
        "split": split,
        "source_id": source or f"source-{name}",
        "file": filename,
        "sha256": digest,
    }


def _write_manifest(root: Path, entries: list[dict]) -> None:
    (root / "dataset_manifest.json").write_text(json.dumps({
        "schema_version": "estimation-calibration-dataset-v1",
        "episodes": entries,
    }))


def test_episode_canonicalizes_trims_and_preserves_default_dtype():
    before = torch.get_default_dtype()
    arrays = _arrays(T=7)
    arrays["row_valid"] = np.array([False, True, True, True, True, True, False])
    episode = _episode(**arrays)
    assert episode.T == 5
    assert episode.N == 3
    assert episode.dt == pytest.approx(0.01)
    assert episode.row_valid.tolist() == [True] * 5
    for value in (episode.time_s, episode.imu, episode.p_BC, episode.gt_R_WB,
                  episode.gt_v_W, episode.gt_p_W):
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert value.is_contiguous()
    assert episode.contact_flags.dtype == torch.bool
    assert torch.get_default_dtype() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("imu", np.zeros((6, 5)), "imu"),
        ("imu", np.zeros((6, 6), dtype=np.int64), "float32 or float64"),
        ("contact_flags", np.ones((6, 3), dtype=np.int8), "bool"),
        ("p_BC", np.zeros((6, 0, 3)), "between 1 and 8"),
        ("p_BC", np.zeros((6, 9, 3)), "between 1 and 8"),
        ("time_s", np.array([0., .01, .02, .019, .04, .05]), "increasing"),
        ("time_s", np.array([0., .01, .02, .031, .04, .05]), "uniformly"),
        ("gt_v_W", np.full((6, 3), np.nan), "finite"),
        ("row_valid", np.array([True, True, False, True, True, True]), "internal hole"),
    ],
)
def test_episode_rejects_invalid_arrays(field, value, message):
    values = _arrays()
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        _episode(**values)


def test_episode_rejects_direct_non_cpu_tensor():
    with pytest.raises(ValueError, match="CPU tensor"):
        _episode(time_s=torch.empty(6, dtype=torch.float64, device="meta"))


@pytest.mark.parametrize("N", [1, 4, 8])
def test_candidate_padding_and_tail_are_inert(N):
    episode = _episode(**_arrays(T=5, N=N))
    rollout = _episode_to_rollout(episode, device=torch.device("cpu"))
    assert rollout.p_BC.shape == (5, 8, 3)
    assert rollout.flags.shape == (5, 8)
    assert not rollout.flags[:, N:].any()
    assert torch.count_nonzero(rollout.p_BC[:, N:]) == 0
    batch = fsi.build_batch([rollout], T_pad=8)
    assert not batch.valid[:, 5:].any()
    assert torch.count_nonzero(batch.dt_row[:, 5:]) == 0
    assert not batch.prop_mask[:, 5:].any()
    assert not batch.correct_mask[:, 5:].any()
    assert not batch.insert_mask[:, 5:].any()


def test_example_is_split_lazy_and_path_free(monkeypatch):
    calls: list[str] = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    dataset = load_dataset("example")
    assert calls == []
    assert dataset.names("train") == ("synthetic-train",)
    assert dataset.load("train")[0].split == "train"
    assert calls == ["train.npz"]
    assert "/home/" not in json.dumps(dataset.identity)


def test_manifest_validates_before_array_load(tmp_path, monkeypatch):
    digest = _write_npz(tmp_path, "first.npz")
    _write_manifest(tmp_path, [
        _entry("first", "train", "first.npz", digest),
        {**_entry("second", "validation", "missing.npz", "0" * 64),
         "extra": True},
    ])
    calls = []
    monkeypatch.setattr(np, "load", lambda *a, **k: calls.append(a) or None)
    with pytest.raises(ValueError, match="unknown or missing"):
        load_dataset(tmp_path)
    assert calls == []


def test_hash_is_checked_before_np_load(tmp_path, monkeypatch):
    digest = _write_npz(tmp_path, "episode.npz")
    _write_manifest(tmp_path, [_entry("episode", "train", "episode.npz", digest)])
    (tmp_path / "episode.npz").write_bytes(b"changed")
    calls = []
    monkeypatch.setattr(np, "load", lambda *a, **k: calls.append(a) or None)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_dataset(tmp_path)
    assert calls == []


@pytest.mark.parametrize("case", ["name", "stem", "hash", "lineage"])
def test_manifest_rejects_global_overlap(tmp_path, case):
    first_hash = _write_npz(tmp_path, "first.npz")
    second_hash = _write_npz(tmp_path, "second.npz", _arrays(T=7))
    first = _entry("first", "train", "first.npz", first_hash, "recording")
    second = _entry("second", "validation", "second.npz", second_hash, "other")
    if case == "name":
        second["name"] = "first"
    elif case == "stem":
        second["file"] = "first.npz"
        second["sha256"] = first_hash
    elif case == "hash":
        shutil.copyfile(tmp_path / "first.npz", tmp_path / "second.npz")
        second["sha256"] = first_hash
    else:
        second["source_id"] = "recording"
    _write_manifest(tmp_path, [first, second])
    with pytest.raises(ValueError, match="duplicate|overlaps"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_npz_keys_are_exact(tmp_path, mutation):
    arrays = _arrays()
    if mutation == "missing":
        arrays.pop("imu")
    else:
        arrays["extra"] = np.zeros(1, dtype=np.float64)
    digest = _write_npz(tmp_path, "episode.npz", arrays)
    _write_manifest(tmp_path, [_entry("episode", "train", "episode.npz", digest)])
    dataset = load_dataset(tmp_path)
    with pytest.raises(ValueError, match="NPZ keys"):
        dataset.load("train")


def test_legacy_adapter_uses_local_basenames(tmp_path):
    arrays = _arrays(T=5, N=2)
    main = {
        "meta/sim_time": arrays["time_s"],
        "input/imu_gyro": arrays["imu"][:, :3],
        "input/imu_accelerometer_native": arrays["imu"][:, 3:],
        "gt/gt_R_WB": arrays["gt_R_WB"],
        "gt/base_linear_velocity": arrays["gt_v_W"],
        "gt/base_position": arrays["gt_p_W"],
    }
    np.savez(tmp_path / "walk.npz", **main)
    np.savez(tmp_path / "walk.features.npz", **{
        "input_kinematics/p_BC": arrays["p_BC"],
        "input_kinematics/v_BC": np.zeros_like(arrays["p_BC"]),
    })
    (tmp_path / "dataset_manifest.json").write_text(json.dumps([{
        "dataset_path": "/private/machine/walk.npz",
        "split": "val",
    }]))
    dataset = load_dataset(tmp_path)
    episode = dataset.load("validation")[0]
    assert episode.name == "walk"
    assert episode.split == "validation"
    assert episode.N == 2
    assert "/private/" not in json.dumps(dataset.identity)
