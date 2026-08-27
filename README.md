# mojo-uncertainties

`mojo-uncertainties` is a standalone Mojo port of the arithmetic core of the
Python [`uncertainties`](https://pythonhosted.org/uncertainties/) package. It
propagates standard deviations through first-order affine derivatives and
preserves correlations when expressions share input variables.

Scalar expressions use a Python API compatible with the covered upstream
names. Large elementwise arrays use compiled Mojo kernels instead of NumPy
object-array loops.

## Coverage

The scalar API covers:

- `ufloat()`, `ufloat_fromstr()`, `nominal_value()`, and `std_dev()`;
- `UFloat`, `Variable`, mutable source standard deviations, derivative maps,
  error components, and standard scores;
- `+`, `-`, `*`, `/`, `**`, reflected operators, unary `+`/`-`, `abs()`, and
  nominal-value comparisons;
- `covariance_matrix()`, `correlation_matrix()`, `correlated_values()`, and
  `correlated_values_norm()`;
- `wrap()` with analytic or numerical partial derivatives;
- the arithmetic-oriented `umath` functions: trigonometric, inverse
  trigonometric, hyperbolic, exponential, logarithmic, square root, error
  functions, `atan2`, `hypot`, `copysign`, `fmod`, `gamma`, `lgamma`, `fsum`,
  `modf`, `frexp`, and `ldexp`.

The `mojo_uncertainties.unumpy` API covers `uarray()`, `nominal_values()`,
`std_devs()`, elementwise arithmetic, broadcasting, slicing, reshape,
transpose, common NumPy ufunc dispatch, and the corresponding arithmetic and
mathematical functions. Shared variable identities survive array expressions
and reordered slices, so `x - x` has exactly zero uncertainty. `gamma()` and
`lgamma()` currently materialize object arrays; the other continuous functions
listed in `unumpy` use Mojo kernels.

This is intentionally not the whole upstream package. The following are not
covered:

- `unumpy.ulinalg`, matrix multiplication, reductions, and generalized
  cross-element transforms;
- in-place array operators, ufunc `out=`, and arbitrary NumPy ufuncs;
- upstream's complete uncertainty-specific formatting mini-language;
- complex values and higher-order or asymmetric uncertainty models.

`uarray()` returns a compact `UArray` rather than an `ndarray(dtype=object)`.
For the covered operations it has the same nominal values, propagated standard
deviations, broadcasting behavior, and correlations. Call
`to_object_array()` when an actual object array is required.

## Install and verify

The repository pins the tested Mojo nightly in `pixi.toml`.

```bash
pixi install
pixi run build
pixi run test
```

The build writes `dist/libmojo-uncertainties.so`. The parity suite compares
against the real `uncertainties` 3.x package installed by Pixi.

## Usage

```python
import numpy as np

from mojo_uncertainties import covariance_matrix, ufloat
from mojo_uncertainties import unumpy as unp

length = ufloat(2.0, 0.03, tag="length")
width = ufloat(4.0, 0.05, tag="width")
area = length * width

print(area.nominal_value, area.std_dev)
print(covariance_matrix([length, area]))

x = unp.uarray(np.array([1.0, 2.0, 3.0]), np.array([0.01, 0.02, 0.03]))
y = (x * x + unp.sin(x)) / 2
print(unp.nominal_values(y))
print(unp.std_devs(y))
print(unp.std_devs(x - x))  # [0. 0. 0.]
```

Run the example after `pixi run build` with `pixi run python`.

## Benchmarks

These are real best-of-three wall-clock results from `pixi run bench`. The
Pixi task holds `/tmp/mojo-bench.lock` for the entire run.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, 72 logical CPUs  
Software: Python 3.13.14, NumPy 2.5.1, uncertainties 3.2.3

| Case | mojo-uncertainties | uncertainties | Upstream / Mojo | Result |
|---|---:|---:|---:|---|
| uarray construction, 200k | 2.571 ms | 392.866 ms | 152.78x | faster |
| independent array addition, 500k | 7.538 ms | 2370.670 ms | 314.50x | faster |
| sin propagation, 500k | 26.568 ms | 2028.740 ms | 76.36x | faster |
| correlated chained expression, 200k | 30.580 ms | 5346.972 ms | 174.85x | faster |
| std_devs after x - reversed(x), 500k | 5.848 ms | 994.193 ms | 170.00x | faster |
| scalar correlated arithmetic loop, 50k | 95.479 ms | 567.800 ms | 5.95x | faster |

The large gains come from replacing upstream's per-element Python objects and
operator calls with contiguous numeric buffers. Scalar work remains
Python-bound, so its gain is much smaller. The scalar arithmetic fast path
avoids generic callable dispatch, temporary constant objects, and redundant
derivative-map copies.

No GPU path is included. The array kernels are streaming passes with only a
few arithmetic operations relative to the data moved. On this machine the
CPU kernels are already 76.36x--314.50x faster than upstream for the array
cases in this benchmark.

## How it works

Each scalar affine value stores a nominal `float` and a sparse mapping from
independent `Variable` objects to first derivatives. Arithmetic applies the
chain rule to those maps. A standard deviation is the root-sum-square of each
derivative times its source standard deviation; covariance uses the shared
keys, which is why correlations and cancellations remain exact.

`UArray` stores C-contiguous `float64` nominal, standard-deviation, and
gradient buffers plus `int64` variable-identity buffers. A single expression
can contain several derivative terms. Slices retain the identities, and the
variance kernel adds covariance terms only where two identities match.
Complex inputs and values that cannot be represented safely as `float64` are
rejected instead of silently narrowed.

Python owns every allocation. The ctypes layer passes buffers across the C ABI
as integer addresses, and the Mojo side reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values. Mojo computes nominal values,
local partial derivatives, chain-rule gradients, and final variances in one
shared library compilation unit. No Python object crosses the ABI and Mojo
does not allocate or retain caller memory.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark must be run through the Pixi task so the machine-wide lock is
held.
