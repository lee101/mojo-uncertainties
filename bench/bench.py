"""Benchmark Mojo array propagation against upstream uncertainties."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "python",
    ),
)

import uncertainties
from uncertainties import umath as upstream_math
from uncertainties import unumpy as upstream

import mojo_uncertainties
from mojo_uncertainties import umath as mojo_math
from mojo_uncertainties import unumpy as mojo


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("benchmark function returned no result")
        best = min(best, elapsed)
    return best


def data(count, seed=0):
    rng = np.random.default_rng(seed)
    nominal = np.ascontiguousarray(rng.uniform(0.5, 2.0, count))
    sigma = np.ascontiguousarray(rng.uniform(0.001, 0.05, count))
    return nominal, sigma


CASES = []


def case(name):
    def decorator(builder):
        CASES.append((name, builder))
        return builder

    return decorator


@case("uarray construction, 200k")
def build_construction():
    nominal, sigma = data(200_000)
    return (
        lambda: mojo.uarray(nominal, sigma),
        lambda: upstream.uarray(nominal, sigma),
    )


@case("independent array addition, 500k")
def build_addition():
    nominal_a, sigma_a = data(500_000)
    nominal_b, sigma_b = data(500_000, seed=1)
    ours_a = mojo.uarray(nominal_a, sigma_a)
    ours_b = mojo.uarray(nominal_b, sigma_b)
    theirs_a = upstream.uarray(nominal_a, sigma_a)
    theirs_b = upstream.uarray(nominal_b, sigma_b)
    return (lambda: ours_a + ours_b, lambda: theirs_a + theirs_b)


@case("sin propagation, 500k")
def build_sin():
    nominal, sigma = data(500_000)
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    return (lambda: mojo.sin(ours), lambda: upstream.sin(theirs))


@case("correlated chained expression, 200k")
def build_expression():
    nominal_a, sigma_a = data(200_000)
    nominal_b, sigma_b = data(200_000, seed=2)
    ours_a = mojo.uarray(nominal_a, sigma_a)
    ours_b = mojo.uarray(nominal_b, sigma_b)
    theirs_a = upstream.uarray(nominal_a, sigma_a)
    theirs_b = upstream.uarray(nominal_b, sigma_b)
    return (
        lambda: (ours_a * ours_b + mojo.sin(ours_a)) / mojo.sqrt(ours_b),
        lambda: (theirs_a * theirs_b + upstream.sin(theirs_a))
        / upstream.sqrt(theirs_b),
    )


@case("std_devs after x - reversed(x), 500k")
def build_std_devs():
    nominal, sigma = data(500_000)
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    ours_expression = ours - ours[::-1]
    their_expression = theirs - theirs[::-1]
    return (
        lambda: mojo.std_devs(ours_expression),
        lambda: upstream.std_devs(their_expression),
    )


@case("scalar correlated arithmetic loop, 50k")
def build_scalar_loop():
    def ours():
        total = mojo_uncertainties.ufloat(1.0, 0.01)
        value = mojo_uncertainties.ufloat(1.2, 0.02)
        for _ in range(50_000):
            total = total + value * value - value
        return total

    def theirs():
        total = uncertainties.ufloat(1.0, 0.01)
        value = uncertainties.ufloat(1.2, 0.02)
        for _ in range(50_000):
            total = total + value * value - value
        return total

    return ours, theirs


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    mojo.uarray([1.0], [0.1]) + 1.0
    print(f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs")
    print(
        f"Software: Python {platform.python_version()}, NumPy {np.__version__}, "
        f"uncertainties {uncertainties.__version__}"
    )
    print()
    print("| Case | mojo-uncertainties | uncertainties | Upstream / Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, builder in CASES:
        ours, theirs = builder()
        ours_time = timeit(ours)
        their_time = timeit(theirs)
        ratio = their_time / ours_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours_time * 1e3:.3f} ms | {their_time * 1e3:.3f} ms "
            f"| {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
