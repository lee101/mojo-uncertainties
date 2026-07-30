"""Load the Mojo shared library and declare its ctypes ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_UNCERTAINTIES_LIB") or os.path.join(
    ROOT, "dist", "libmojo-uncertainties.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "munc_binary": ([I, I, I, I, I, I, I], None),
    "munc_unary": ([I, I, I, I, I], None),
    "munc_chain_binary": ([I, I, I, I, I, I], None),
    "munc_chain_unary": ([I, I, I, I], None),
    "munc_variance_diag": ([I, I, I, I], None),
    "munc_variance_cross": ([I, I, I, I, I, I, I, I], None),
    "munc_std_finish": ([I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_UNCERTAINTIES_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_UNCERTAINTIES_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_UNCERTAINTIES_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(directory, filename)
        for directory, _, filenames in os.walk(SRC)
        for filename in filenames
        if filename.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(source) for source in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        os.path.join(SRC, "kernels.mojo"),
        "-o",
        LIB,
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if process.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value) -> np.ndarray:
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values are not supported")
    if np.issubdtype(source.dtype, np.floating) and source.dtype.itemsize > 8:
        raise TypeError("floating-point values wider than float64 are not supported")
    if np.issubdtype(source.dtype, np.integer) and source.size:
        limit = 2**53
        if np.any(source > limit) or np.any(source < -limit):
            raise OverflowError("integer values outside the exact float64 range are not supported")
    return np.ascontiguousarray(source, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError(f"FFI buffer has unsupported dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffer must be C-contiguous")
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("FFI buffer has a null pointer")
    return address


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
