"""Validated, split-aware calibration datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


_SCHEMA = "estimation-calibration-dataset-v1"
_SPLITS = ("train", "validation", "test")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_ARRAYS = {
    "time_s",
    "imu",
    "p_BC",
    "contact_flags",
    "gt_R_WB",
    "gt_v_W",
    "gt_p_W",
}


def _numeric(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have float32 or float64 dtype")
        return value.detach().to(torch.float64).contiguous().clone()
    array = np.asarray(value)
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"{name} must have float32 or float64 dtype")
    return torch.from_numpy(np.array(array, dtype=np.float64, order="C", copy=True))


def _boolean(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have bool dtype")
        return value.detach().contiguous().clone()
    array = np.asarray(value)
    if array.dtype != np.dtype("bool"):
        raise TypeError(f"{name} must have bool dtype")
    return torch.from_numpy(np.array(array, dtype=bool, order="C", copy=True))


@dataclass(frozen=True)
class CalibrationEpisode:
    """One portable, CPU-resident calibration episode."""

    name: str
    split: str
    source_id: str
    time_s: torch.Tensor
    imu: torch.Tensor
    p_BC: torch.Tensor
    contact_flags: torch.Tensor
    gt_R_WB: torch.Tensor
    gt_v_W: torch.Tensor
    gt_p_W: torch.Tensor
    row_valid: torch.Tensor | None = None
    _dt: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("name is not a portable identifier")
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        if not isinstance(self.source_id, str) or not _IDENTIFIER.fullmatch(self.source_id):
            raise ValueError("source_id is not a portable identifier")

        numeric = {
            key: _numeric(getattr(self, key), key)
            for key in ("time_s", "imu", "p_BC", "gt_R_WB", "gt_v_W", "gt_p_W")
        }
        flags = _boolean(self.contact_flags, "contact_flags")
        time_s = numeric["time_s"]
        if time_s.ndim != 1 or time_s.numel() < 2:
            raise ValueError("time_s must have shape [T] with T >= 2")
        T = time_s.shape[0]
        if numeric["imu"].shape != (T, 6):
            raise ValueError("imu must have shape [T, 6]")
        if numeric["p_BC"].ndim != 3 or numeric["p_BC"].shape[0] != T \
                or numeric["p_BC"].shape[2] != 3:
            raise ValueError("p_BC must have shape [T, N, 3]")
        N = numeric["p_BC"].shape[1]
        if not 1 <= N <= 8:
            raise ValueError("p_BC candidate count must be between 1 and 8")
        if flags.shape != (T, N):
            raise ValueError("contact_flags must have shape [T, N]")
        expected = {
            "gt_R_WB": (T, 3, 3),
            "gt_v_W": (T, 3),
            "gt_p_W": (T, 3),
        }
        for key, shape in expected.items():
            if numeric[key].shape != shape:
                raise ValueError(f"{key} must have shape {list(shape)}")
        if any(not torch.isfinite(value).all() for value in numeric.values()):
            raise ValueError("numeric episode arrays must be finite")

        valid = (torch.ones(T, dtype=torch.bool) if self.row_valid is None
                 else _boolean(self.row_valid, "row_valid"))
        if valid.shape != (T,):
            raise ValueError("row_valid must have shape [T]")
        rows = torch.nonzero(valid, as_tuple=True)[0]
        if rows.numel() < 2:
            raise ValueError("row_valid must retain at least two rows")
        first, last = int(rows[0]), int(rows[-1])
        if int(valid[first:last + 1].sum()) != last - first + 1:
            raise ValueError("row_valid contains an internal hole; split the episode")

        physical = slice(first, last + 1)
        for key, value in numeric.items():
            value = value[physical].contiguous().clone()
            object.__setattr__(self, key, value)
        object.__setattr__(self, "contact_flags", flags[physical].contiguous().clone())
        object.__setattr__(self, "row_valid", torch.ones(last - first + 1,
                                                         dtype=torch.bool))

        deltas = torch.diff(self.time_s)
        if not torch.all(deltas > 0):
            raise ValueError("time_s must be strictly increasing")
        dt = float(np.median(deltas.numpy()))
        tolerance = max(1e-9, 1e-3 * dt)
        if not torch.all(torch.abs(deltas - dt) <= tolerance):
            raise ValueError("time_s must be uniformly sampled")
        object.__setattr__(self, "_dt", dt)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def T(self) -> int:
        return self.time_s.shape[0]

    @property
    def N(self) -> int:
        return self.p_BC.shape[1]


@dataclass(frozen=True)
class _EpisodeRecord:
    name: str
    split: str
    source_id: str
    file: Path
    sha256: str
    format: str = "v1"
    feature_file: Path | None = None


class _CalibrationDataset:
    def __init__(self, records: tuple[_EpisodeRecord, ...], schema: str) -> None:
        self._records = records
        self._schema = schema
        self._cache: dict[str, tuple[CalibrationEpisode, ...]] = {}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": self._schema,
            "episodes": [
                {
                    "name": record.name,
                    "split": record.split,
                    "source_id": record.source_id,
                    "file": record.file.name,
                    "sha256": record.sha256,
                }
                for record in self._records
            ],
        }

    def names(self, split: str | None = None) -> tuple[str, ...]:
        if split is not None and split not in _SPLITS:
            raise ValueError("unknown dataset split")
        return tuple(record.name for record in self._records
                     if split is None or record.split == split)

    def load(self, split: str) -> tuple[CalibrationEpisode, ...]:
        if split not in _SPLITS:
            raise ValueError("unknown dataset split")
        if split not in self._cache:
            self._cache[split] = tuple(
                _load_record(record) for record in self._records
                if record.split == split
            )
        return self._cache[split]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_manifest(root: Path) -> Any:
    manifest = root / "dataset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("dataset_manifest.json not found")
    try:
        return json.loads(manifest.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("dataset manifest is not valid JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_pair(first: Path, second: Path) -> str:
    digest = hashlib.sha256()
    for label, path in ((b"data\0", first), (b"features\0", second)):
        digest.update(label)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _v1_index(root: Path, manifest: dict[str, Any]) -> _CalibrationDataset:
    if set(manifest) != {"schema_version", "episodes"}:
        raise ValueError("dataset manifest has unknown or missing fields")
    if manifest["schema_version"] != _SCHEMA:
        raise ValueError("unsupported dataset schema_version")
    if not isinstance(manifest["episodes"], list):
        raise TypeError("manifest episodes must be a list")

    root = root.resolve()
    records: list[_EpisodeRecord] = []
    names: set[str] = set()
    stems: set[str] = set()
    files: set[Path] = set()
    hashes: set[str] = set()
    lineage: dict[str, str] = {}
    for entry in manifest["episodes"]:
        if not isinstance(entry, dict) or set(entry) != {
                "name", "split", "source_id", "file", "sha256"}:
            raise ValueError("episode manifest entry has unknown or missing fields")
        name, split, source_id = entry["name"], entry["split"], entry["source_id"]
        filename, declared_hash = entry["file"], entry["sha256"]
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("episode name is not a portable identifier")
        if split not in _SPLITS:
            raise ValueError("episode split must be train, validation, or test")
        if not isinstance(source_id, str) or not _IDENTIFIER.fullmatch(source_id):
            raise ValueError("episode source_id is not a portable identifier")
        if not isinstance(filename, str) or Path(filename).name != filename \
                or not filename.endswith(".npz"):
            raise ValueError("episode file must be a relative .npz basename")
        if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash):
            raise ValueError("episode sha256 must be 64 lowercase hex characters")
        path = root / filename
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"episode file not found: {filename}") from error
        if resolved.parent != root or not resolved.is_file():
            raise ValueError("episode file escapes the dataset directory")
        stem = Path(filename).stem
        if name in names:
            raise ValueError("duplicate episode name")
        if stem in stems:
            raise ValueError("duplicate episode file stem")
        if resolved in files:
            raise ValueError("duplicate episode file")
        if declared_hash in hashes:
            raise ValueError("duplicate episode content hash")
        if source_id in lineage and lineage[source_id] != split:
            raise ValueError("source_id overlaps dataset splits")
        names.add(name)
        stems.add(stem)
        files.add(resolved)
        hashes.add(declared_hash)
        lineage[source_id] = split
        records.append(_EpisodeRecord(name, split, source_id, resolved, declared_hash))

    for record in records:
        if _sha256_file(record.file) != record.sha256:
            raise ValueError(f"episode sha256 mismatch: {record.file.name}")
    return _CalibrationDataset(tuple(records), _SCHEMA)


def _load_g1_paired_index(root: Path, manifest: list[Any]) -> _CalibrationDataset:
    """Index the historical paired G1 bundle without preserving old paths."""
    root = root.resolve()
    records: list[_EpisodeRecord] = []
    names: set[str] = set()
    files: set[Path] = set()
    hashes: set[str] = set()
    lineage: dict[str, str] = {}
    for entry in manifest:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset_path"), str) \
                or not isinstance(entry.get("split"), str):
            raise ValueError("legacy manifest entry is not a paired G1 record")
        stem = Path(entry["dataset_path"]).stem
        if not _IDENTIFIER.fullmatch(stem):
            raise ValueError("legacy episode stem is not portable")
        split = {"val": "validation"}.get(entry["split"], entry["split"])
        if split not in _SPLITS:
            raise ValueError("legacy episode split is invalid")
        data_file = (root / f"{stem}.npz").resolve()
        feature_file = (root / f"{stem}.features.npz").resolve()
        if data_file.parent != root or feature_file.parent != root \
                or not data_file.is_file() or not feature_file.is_file():
            raise FileNotFoundError(f"legacy episode pair is incomplete: {stem}")
        if stem in names or data_file in files or feature_file in files:
            raise ValueError("duplicate legacy episode")
        source_id = stem
        if source_id in lineage and lineage[source_id] != split:
            raise ValueError("legacy source_id overlaps dataset splits")
        names.add(stem)
        files.update((data_file, feature_file))
        lineage[source_id] = split
        content_hash = _sha256_pair(data_file, feature_file)
        if content_hash in hashes:
            raise ValueError("duplicate legacy episode content hash")
        hashes.add(content_hash)
        records.append(_EpisodeRecord(stem, split, source_id, data_file,
                                      content_hash, "g1-paired", feature_file))
    return _CalibrationDataset(tuple(records), "g1-paired-compat-v1")


def _load_record(record: _EpisodeRecord) -> CalibrationEpisode:
    if record.format == "g1-paired":
        return _load_g1_pair(record)
    with np.load(record.file, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != _REQUIRED_ARRAYS and keys != _REQUIRED_ARRAYS | {"row_valid"}:
            raise ValueError(f"episode NPZ keys are invalid: {record.file.name}")
        arrays = {key: archive[key].copy() for key in archive.files}
    return CalibrationEpisode(
        name=record.name,
        split=record.split,
        source_id=record.source_id,
        row_valid=arrays.get("row_valid"),
        **{key: arrays[key] for key in _REQUIRED_ARRAYS},
    )


def _load_g1_pair(record: _EpisodeRecord) -> CalibrationEpisode:
    if record.feature_file is None:
        raise ValueError("legacy feature file is missing")
    main_keys = {
        "meta/sim_time", "input/imu_gyro", "input/imu_accelerometer_native",
        "gt/gt_R_WB", "gt/base_linear_velocity", "gt/base_position",
    }
    feature_keys = {"input_kinematics/p_BC", "input_kinematics/v_BC"}
    with np.load(record.file, allow_pickle=False) as main:
        if not main_keys <= set(main.files):
            raise ValueError(f"legacy episode arrays are incomplete: {record.name}")
        data = {key: main[key].copy() for key in main_keys}
    with np.load(record.feature_file, allow_pickle=False) as features:
        if not feature_keys <= set(features.files):
            raise ValueError(f"legacy feature arrays are incomplete: {record.name}")
        p_BC = features["input_kinematics/p_BC"].copy()
        v_BC = features["input_kinematics/v_BC"].copy()
    from .covariance_calibration import candidate_reliability, hysteresis_contact_schedule
    flags = hysteresis_contact_schedule(candidate_reliability(p_BC, v_BC))
    imu = np.concatenate((data["input/imu_gyro"],
                          data["input/imu_accelerometer_native"]), axis=1)
    return CalibrationEpisode(
        name=record.name,
        split=record.split,
        source_id=record.source_id,
        time_s=data["meta/sim_time"],
        imu=imu,
        p_BC=p_BC,
        contact_flags=flags,
        gt_R_WB=data["gt/gt_R_WB"],
        gt_v_W=data["gt/base_linear_velocity"],
        gt_p_W=data["gt/base_position"],
    )


def load_dataset(path: str | Path) -> _CalibrationDataset:
    """Validate dataset metadata now and load episode arrays only by split."""
    root = Path(__file__).parent / "_example_data" if path == "example" else Path(path)
    if not root.is_dir():
        raise FileNotFoundError("dataset directory not found")
    manifest = _read_manifest(root)
    if isinstance(manifest, list):
        return _load_g1_paired_index(root, manifest)
    if not isinstance(manifest, dict):
        raise TypeError("dataset manifest must be an object")
    return _v1_index(root, manifest)


def _episode_to_rollout(
    episode: CalibrationEpisode,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    trim_s: float = 0.0,
):
    """Move an episode once and pad its candidate axis for fixed-slot replay."""
    if trim_s < 0:
        raise ValueError("trim_s must be nonnegative")
    n_trim = int(round(trim_s / episode.dt))
    trim0, trim1 = n_trim, episode.T - n_trim
    if trim1 - trim0 < 2:
        raise ValueError(f"episode is too short after trimming: {episode.name}")
    p_BC = torch.zeros(episode.T, 8, 3, dtype=dtype, device=device)
    p_BC[:, :episode.N] = episode.p_BC.to(device=device, dtype=dtype)
    flags = np.zeros((episode.T, 8), dtype=bool)
    flags[:, :episode.N] = episode.contact_flags.numpy()
    imu = episode.imu.to(device=device, dtype=dtype)
    gt_R_WB = episode.gt_R_WB.to(device=device, dtype=dtype)
    gt_v_W = episode.gt_v_W.to(device=device, dtype=dtype)
    gt_p_W = episode.gt_p_W.to(device=device, dtype=dtype)
    from .covariance_calibration import Rollout
    from .invariant_ekf import precompute_contact_changes
    return Rollout(
        stem=episode.name,
        split_label=episode.split,
        dt=episode.dt,
        total_rows=episode.T,
        trim0=trim0,
        trim1=trim1,
        imu=imu,
        p_BC=p_BC,
        flags=flags,
        changes=precompute_contact_changes(flags),
        gt_R_WB=gt_R_WB,
        gt_v_W=gt_v_W,
        gt_v_B=torch.einsum("tji,tj->ti", gt_R_WB, gt_v_W),
        gt_p_W=gt_p_W,
    )
