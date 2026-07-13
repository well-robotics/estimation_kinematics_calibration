"""Small public API for covariance calibration runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch

from .batched_calibration import eval_batched, train_batched
from .covariance_calibration import (
    CalibrationConfig,
    aggregate_metrics,
    build_covs,
    fixed_initial_covariance,
    make_cov_modules,
    train_trimmed_rollouts,
)
from .data import (
    CalibrationEpisode,
    _CalibrationDataset,
    _episode_to_rollout,
    load_dataset,
)


_CHECKPOINT_SCHEMA = "estimation-calibration-checkpoint-v2"
_METRICS_SCHEMA = "estimation-calibration-metrics-v1"
_RUN_SCHEMA = "estimation-calibration-run-v1"
_RUN_FILES = {"checkpoint.pt", "covariances.npz", "metrics.json", "manifest.json"}
_PAYLOAD_FILES = {"checkpoint.pt", "covariances.npz", "metrics.json"}
_COVARIANCE_KEYS = {"Qg", "Qa", "Qbg", "Qba", "Qc", "R_kin_pos"}
_COMPILE_CHOICES = {"auto", "none", "default", "cuda-graph", "cuda-graph-compile"}
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_KEYS = {
    "schema_version", "current_state_dict", "optimizer_state_dict", "next_epoch",
    "completed_epochs", "target_epochs", "best", "history", "validation_curve",
    "chunk_trace", "rng_state", "config_identity", "dataset_identity",
}
_METRICS_KEYS = {
    "schema_version", "selected_epoch",
    "selected_validation_body_velocity_rmse_mps", "train", "validation", "test",
}
_EXECUTION_KEYS = {
    "requested_device", "effective_device", "requested_compile_mode",
    "effective_compile_mode", "fallback_reason_code",
}


@dataclass(frozen=True)
class CalibrationResult:
    run_dir: Path
    selected_epoch: int
    selected_validation_body_velocity_rmse_mps: float
    covariances: Mapping[str, torch.Tensor]
    metrics: Mapping[str, Any]
    artifact_manifest: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, writer) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    _atomic(path, write)


def _write_torch(path: Path, payload: dict) -> None:
    _atomic(path, lambda temporary: torch.save(payload, temporary))


def _write_npz(path: Path, arrays: Mapping[str, torch.Tensor]) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as stream:
            np.savez(stream, **{
                key: value.detach().cpu().numpy() for key, value in arrays.items()
            })
    _atomic(path, write)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _public_metrics(metrics: dict) -> dict:
    hidden = {"body_velocity_sse", "position_sse", "orientation_sum_deg",
              "nis_per_dim_sum"}
    return {key: value for key, value in metrics.items() if key not in hidden}


def _resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if request == "cpu":
        return torch.device("cpu")
    match = _CUDA_DEVICE.fullmatch(request) if isinstance(request, str) else None
    if match is None:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = int(match.group(1) or 0)
    if index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device does not exist")
    return torch.device("cuda", index)


def _resolve_compile(request: str | None, device: torch.device) -> tuple[str, str, str | None]:
    requested = "auto" if request is None else request
    if requested not in _COMPILE_CHOICES:
        raise ValueError("compile must be auto, none, default, cuda-graph, or cuda-graph-compile")
    effective = ("cuda-graph" if device.type == "cuda" else "none") \
        if requested == "auto" else requested
    if effective in ("cuda-graph", "cuda-graph-compile") and device.type != "cuda":
        raise ValueError(f"{effective} requires a CUDA device")
    return requested, effective, None if effective == "none" else effective


def _validate_config(config: CalibrationConfig) -> tuple[torch.device, str, str, str | None]:
    if not isinstance(config, CalibrationConfig):
        raise TypeError("config must be CalibrationConfig")
    if config.dtype != torch.float64:
        raise ValueError("v0.2.0 supports torch.float64 calibration only")
    if not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config.epochs <= 0 or config.chunk <= 0 or config.lr <= 0:
        raise ValueError("epochs, chunk, and lr must be positive")
    if config.trim_s < 0 or config.s_jitter <= 0 or config.bias_lr_factor <= 0 \
            or config.fallback_lr <= 0:
        raise ValueError("numerical and optimizer settings must be positive")
    if config.exec_mode not in ("batched", "sequential"):
        raise ValueError("exec_mode must be batched or sequential")
    device = _resolve_device(config.device)
    requested, effective, internal = _resolve_compile(config.compile_mode, device)
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("this configuration requires CUDA")
    if config.exec_mode == "sequential" and effective != "none":
        raise ValueError("sequential execution supports eager mode only")
    return device, requested, effective, internal


def _config_identity(
    config: CalibrationConfig,
    device: torch.device,
    requested_compile: str,
    effective_compile: str,
    fallback_reason: str | None,
) -> dict:
    return {
        "trim_s": config.trim_s,
        "s_jitter": config.s_jitter,
        "chunk": config.chunk,
        "lr": config.lr,
        "fallback_lr": config.fallback_lr,
        "bias_lr_factor": config.bias_lr_factor,
        "dtype": "float64",
        "seed": config.seed,
        "require_cuda": config.require_cuda,
        "profile_stages": config.profile_stages,
        "requested_device": config.device,
        "effective_device": str(device),
        "requested_exec_mode": config.exec_mode,
        "effective_exec_mode": config.exec_mode,
        "requested_compile_mode": requested_compile,
        "effective_compile_mode": effective_compile,
        "fallback_reason_code": fallback_reason,
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8},
    }


def _split_rollouts(
    dataset: _CalibrationDataset,
    split: str,
    config: CalibrationConfig,
    device: torch.device,
) -> tuple[list[str], dict]:
    episodes = dataset.load(split)
    order = [episode.name for episode in episodes]
    return order, {
        episode.name: _episode_to_rollout(
            episode, device=device, dtype=config.dtype, trim_s=config.trim_s)
        for episode in episodes
    }


def _selected_covariances(best: dict, device: torch.device) -> dict[str, torch.Tensor]:
    model = make_cov_modules(device=device, dtype=torch.float64)
    model.load_state_dict(best["state"])
    with torch.no_grad():
        covariances, measurement = build_covs(model)
    return {
        **{key: value.detach() for key, value in covariances.items()},
        "R_kin_pos": measurement.detach(),
    }


def _metrics_payload(state: dict) -> dict:
    history = state["history"]
    best = state["best"]
    return {
        "schema_version": _METRICS_SCHEMA,
        "selected_epoch": best["epoch"],
        "selected_validation_body_velocity_rmse_mps": best[
            "validation_body_velocity_rmse_mps"],
        "train": [
            {
                "epoch": row["epoch"],
                "body_velocity_mse_m2ps2": row["train_body_loss"],
                "regularization_loss": row["train_reg_loss"],
                "nis_per_measurement_dim": row["nis_per_dim_mean"],
            }
            for row in history
        ],
        "validation": [
            {"epoch": row["epoch"], **_public_metrics(row["validation"])}
            for row in history
        ],
        "test": None,
    }


def _write_manifest(run_dir: Path, execution: dict) -> dict:
    artifacts = {name: _sha256(run_dir / name) for name in sorted(_PAYLOAD_FILES)}
    manifest = {
        "schema_version": _RUN_SCHEMA,
        "artifacts": artifacts,
        "execution": execution,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def _write_generation(
    run_dir: Path,
    state: dict,
    *,
    config: CalibrationConfig,
    device: torch.device,
    requested_compile: str,
    dataset_identity: dict,
) -> None:
    effective = state["effective_compile_mode"]
    fallback = state["fallback_reason_code"]
    identity = _config_identity(config, device, requested_compile, effective, fallback)
    checkpoint = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "current_state_dict": _to_cpu(state["current_state_dict"]),
        "optimizer_state_dict": _to_cpu(state["optimizer_state_dict"]),
        "next_epoch": state["next_epoch"],
        "completed_epochs": state["next_epoch"],
        "target_epochs": config.epochs,
        "best": _to_cpu(state["best"]),
        "history": copy.deepcopy(state["history"]),
        "validation_curve": [
            row["validation"]["body_velocity_rmse_mps"] for row in state["history"]
        ],
        "chunk_trace": list(state["chunk_trace"]),
        "rng_state": _to_cpu(state["rng_state"]),
        "config_identity": identity,
        "dataset_identity": copy.deepcopy(dataset_identity),
    }
    covariances = _selected_covariances(state["best"], device)
    _write_torch(run_dir / "checkpoint.pt", checkpoint)
    _write_npz(run_dir / "covariances.npz", covariances)
    _write_json(run_dir / "metrics.json", _metrics_payload(state))
    _write_manifest(run_dir, {
        "requested_device": config.device,
        "effective_device": str(device),
        "requested_compile_mode": requested_compile,
        "effective_compile_mode": effective,
        "fallback_reason_code": fallback,
    })


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run file: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid run file: {path.name}")
    return value


def _read_run(run_dir: Path) -> tuple[dict, dict, dict, dict[str, torch.Tensor]]:
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != _RUN_FILES:
        raise ValueError("run directory must contain exactly four artifacts")
    manifest = _read_json(run_dir / "manifest.json")
    if set(manifest) != {"schema_version", "artifacts", "execution"} \
            or manifest.get("schema_version") != _RUN_SCHEMA \
            or set(manifest.get("artifacts", {})) != _PAYLOAD_FILES \
            or set(manifest.get("execution", {})) != _EXECUTION_KEYS:
        raise ValueError("invalid run manifest schema")
    for name, expected in manifest["artifacts"].items():
        if not isinstance(expected, str) or not _DIGEST.fullmatch(expected) \
                or _sha256(run_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                            weights_only=False)
    metrics = _read_json(run_dir / "metrics.json")
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS \
            or checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError("invalid checkpoint schema")
    if set(metrics) != _METRICS_KEYS or metrics.get("schema_version") != _METRICS_SCHEMA:
        raise ValueError("invalid metrics schema")
    completed = checkpoint["completed_epochs"]
    if checkpoint["next_epoch"] != completed or completed != len(checkpoint["history"]) \
            or completed != len(metrics["train"]) or completed != len(metrics["validation"]):
        raise ValueError("inconsistent checkpoint epoch state")
    best = checkpoint["best"]
    if metrics["selected_epoch"] != best["epoch"] \
            or metrics["selected_validation_body_velocity_rmse_mps"] != best[
                "validation_body_velocity_rmse_mps"]:
        raise ValueError("inconsistent selected checkpoint state")
    with np.load(run_dir / "covariances.npz", allow_pickle=False) as archive:
        if set(archive.files) != _COVARIANCE_KEYS:
            raise ValueError("invalid covariance artifact schema")
        covariances = {
            key: torch.from_numpy(np.array(archive[key], dtype=np.float64, copy=True))
            for key in archive.files
        }
    if any(value.shape != (3, 3) or not torch.isfinite(value).all()
           for value in covariances.values()):
        raise ValueError("invalid covariance artifact values")
    return manifest, checkpoint, metrics, covariances


def _result(run_dir: Path) -> CalibrationResult:
    manifest, _, metrics, covariances = _read_run(run_dir)
    return CalibrationResult(
        run_dir=run_dir,
        selected_epoch=int(metrics["selected_epoch"]),
        selected_validation_body_velocity_rmse_mps=float(
            metrics["selected_validation_body_velocity_rmse_mps"]),
        covariances=MappingProxyType(covariances),
        metrics=MappingProxyType(metrics),
        artifact_manifest=MappingProxyType(dict(manifest["artifacts"])),
    )


def calibrate(
    dataset: _CalibrationDataset,
    config: CalibrationConfig,
    *,
    output_dir: str | Path,
    resume: bool = False,
) -> CalibrationResult:
    """Calibrate on train, select on validation, and write four run files."""
    if not isinstance(dataset, _CalibrationDataset):
        raise TypeError("dataset must be returned by load_dataset")
    device, requested_compile, predicted_compile, internal_compile = _validate_config(config)
    run_dir = Path(output_dir)
    resume_state = None
    if resume:
        manifest, checkpoint, metrics, _ = _read_run(run_dir)
        if metrics["test"] is not None:
            raise ValueError("a test-evaluated run cannot be resumed")
        if checkpoint["dataset_identity"] != dataset.identity:
            raise ValueError("resume dataset identity mismatch")
        completed = int(checkpoint["completed_epochs"])
        previous_target = int(checkpoint["target_epochs"])
        if config.epochs < previous_target or config.epochs <= completed:
            raise ValueError("resume epochs must preserve or extend the target")
        stored_identity = checkpoint["config_identity"]
        stored_effective = stored_identity["effective_compile_mode"]
        stored_fallback = stored_identity["fallback_reason_code"]
        candidate_identity = _config_identity(
            config, device, requested_compile, stored_effective, stored_fallback)
        if candidate_identity != stored_identity:
            raise ValueError("resume configuration identity mismatch")
        if stored_fallback is None and stored_effective != predicted_compile:
            raise ValueError("resume execution identity mismatch")
        internal_compile = None if stored_effective == "none" else stored_effective
        resume_state = {
            key: checkpoint[key] for key in (
                "current_state_dict", "optimizer_state_dict", "next_epoch",
                "history", "chunk_trace", "best", "rng_state")
        }
        resume_state["effective_compile_mode"] = stored_effective
        resume_state["fallback_reason_code"] = stored_fallback
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("output directory is not empty; pass resume=True")
        run_dir.mkdir(parents=True, exist_ok=True)

    train_order, train_rollouts = _split_rollouts(dataset, "train", config, device)
    validation_order, validation_rollouts = _split_rollouts(
        dataset, "validation", config, device)
    if not train_order or not validation_order:
        raise ValueError("calibration requires nonempty train and validation splits")
    internal_config = replace(config, compile_mode=internal_compile)

    def save_epoch(state: dict) -> None:
        _write_generation(
            run_dir, state, config=config, device=device,
            requested_compile=requested_compile,
            dataset_identity=dataset.identity,
        )

    trainer = train_batched if config.exec_mode == "batched" else train_trimmed_rollouts
    trainer(
        train_order, train_rollouts,
        validation_order=validation_order,
        validation_rollouts=validation_rollouts,
        config=internal_config,
        device=device,
        resume_state=resume_state,
        epoch_callback=save_epoch,
    )
    return _result(run_dir)


def evaluate(
    dataset: _CalibrationDataset,
    *,
    checkpoint: str | Path,
    split: str = "test",
    device: str = "auto",
) -> Mapping[str, Any]:
    """Evaluate the frozen validation-selected state on test exactly once."""
    if split != "test":
        raise ValueError("v0.2.0 evaluate accepts split='test' only")
    if not isinstance(dataset, _CalibrationDataset):
        raise TypeError("dataset must be returned by load_dataset")
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.name != "checkpoint.pt":
        raise ValueError("checkpoint must name checkpoint.pt")
    run_dir = checkpoint_path.parent
    manifest, saved, metrics, _ = _read_run(run_dir)
    if metrics["test"] is not None:
        raise ValueError("test metrics have already been written for this run")
    if saved["dataset_identity"] != dataset.identity:
        raise ValueError("evaluation dataset identity mismatch")
    target_device = _resolve_device(device)

    identity = saved["config_identity"]
    eval_config = CalibrationConfig(
        trim_s=identity["trim_s"],
        s_jitter=identity["s_jitter"],
        chunk=identity["chunk"],
        dtype=torch.float64,
        device=device,
        require_cuda=False,
        compile_mode="none",
        seed=identity["seed"],
    )
    order, rollouts = _split_rollouts(dataset, "test", eval_config, target_device)
    if not order:
        raise ValueError("evaluation requires a nonempty test split")
    model = make_cov_modules(device=target_device, dtype=torch.float64)
    model.load_state_dict(saved["best"]["state"])
    with torch.no_grad():
        covariances, measurement = build_covs(model)
        per_episode = eval_batched(
            order, rollouts, covariances, measurement,
            P0_fixed=fixed_initial_covariance(target_device),
            s_jitter=eval_config.s_jitter,
        )
    aggregate = aggregate_metrics(per_episode)
    if not aggregate["finite"] or aggregate["final_P_sym"] >= 1e-9 \
            or aggregate["final_P_min_eig"] <= -1e-12:
        raise FloatingPointError("test replay failed final covariance checks")
    test_metrics = {
        "aggregate": _public_metrics(aggregate),
        "episodes": {name: _public_metrics(value)
                     for name, value in per_episode.items()},
    }
    metrics["test"] = test_metrics
    _write_json(run_dir / "metrics.json", metrics)
    _write_manifest(run_dir, manifest["execution"])
    return MappingProxyType(test_metrics)


def inspect_run(run_dir: str | Path) -> dict:
    manifest, checkpoint, metrics, _ = _read_run(Path(run_dir))
    return {
        "completed_epochs": checkpoint["completed_epochs"],
        "selected_epoch": metrics["selected_epoch"],
        "test_evaluated": metrics["test"] is not None,
        "execution": manifest["execution"],
        "artifacts": manifest["artifacts"],
    }
