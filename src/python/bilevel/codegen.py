"""Loading generated CasADi external functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from casadi import external

def _pick_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "none of the generated code libraries exist: "
        + ", ".join(str(path) for path in paths)
    )


@dataclass(frozen=True)
class CodegenFunctions:
    """Generated kinematic derivative functions."""

    foot_velocity: object
    foot_position: object


class CodegenLibraryLoader:
    def __init__(self, library_dir: str | Path):
        self.library_dir = Path(library_dir)

    def load(self) -> CodegenFunctions:
        lib_v = _pick_first_existing(
            [
                self.library_dir / "libyv_and_J_codegen.so",
                self.library_dir / "libyv_and_J_codegen.dylib",
                self.library_dir / "libyv_and_J_codegen.dll",
            ]
        )
        lib_p = _pick_first_existing(
            [
                self.library_dir / "libpf_and_J_codegen.so",
                self.library_dir / "libpf_and_J_codegen.dylib",
                self.library_dir / "libpf_and_J_codegen.dll",
            ]
        )
        return CodegenFunctions(
            foot_velocity=external("yv_and_J", str(lib_v)),
            foot_position=external("pf_and_J", str(lib_p)),
        )
