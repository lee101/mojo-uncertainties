"""Mojo-backed elementwise arrays of affine uncertain values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import umath
from .._lib import addr, f64, i64, lib
from ..core import AffineScalarFunc, Variable, nominal_value, std_dev


_BINARY_OPS = {
    "add": 0,
    "subtract": 1,
    "multiply": 2,
    "divide": 3,
    "power": 4,
    "atan2": 5,
    "hypot": 6,
    "copysign": 7,
    "fmod": 8,
}
_UNARY_OPS = {
    "negative": 0,
    "positive": 1,
    "absolute": 2,
    "exp": 3,
    "expm1": 4,
    "log": 5,
    "log10": 6,
    "log1p": 7,
    "sqrt": 8,
    "sin": 9,
    "cos": 10,
    "tan": 11,
    "asin": 12,
    "acos": 13,
    "atan": 14,
    "sinh": 15,
    "cosh": 16,
    "tanh": 17,
    "asinh": 18,
    "acosh": 19,
    "atanh": 20,
    "erf": 21,
    "erfc": 22,
    "degrees": 23,
    "radians": 24,
}

_next_variable_id = 1
_scalar_ids: dict[Variable, int] = {}
_variables_by_id: dict[int, Variable] = {}


def _allocate_ids(count: int) -> np.ndarray:
    global _next_variable_id
    start = _next_variable_id
    _next_variable_id += count
    return np.arange(start, start + count, dtype=np.int64)


def _id_for_variable(variable: Variable) -> int:
    global _next_variable_id
    identifier = _scalar_ids.get(variable)
    if identifier is None:
        identifier = _next_variable_id
        _next_variable_id += 1
        _scalar_ids[variable] = identifier
        _variables_by_id[identifier] = variable
    return identifier


def _variable_for_id(identifier: int, sigma: float) -> Variable:
    variable = _variables_by_id.get(identifier)
    if variable is None:
        variable = Variable(0.0, sigma)
        _variables_by_id[identifier] = variable
    return variable


@dataclass(frozen=True)
class _Term:
    ids: np.ndarray
    sigma: np.ndarray
    gradient: np.ndarray


def _broadcast(array: np.ndarray, shape: tuple[int, ...], dtype) -> np.ndarray:
    return np.ascontiguousarray(np.broadcast_to(array, shape), dtype=dtype)


def _term_for_shape(term: _Term, shape: tuple[int, ...]) -> _Term:
    return _Term(
        _broadcast(term.ids, shape, np.int64),
        _broadcast(term.sigma, shape, np.float64),
        _broadcast(term.gradient, shape, np.float64),
    )


class UArray:
    """An ndarray-like collection with independent affine variables per element."""

    __array_priority__ = 10000

    def __init__(self, nominal: np.ndarray, terms: tuple[_Term, ...] = ()):
        self._nominal = f64(nominal)
        self._terms = tuple(
            _Term(i64(term.ids), f64(term.sigma), f64(term.gradient)) for term in terms
        )

    @classmethod
    def independent(cls, nominal, sigma) -> "UArray":
        nominal_array, sigma_array = np.broadcast_arrays(
            f64(nominal),
            f64(sigma),
        )
        nominal_array = f64(nominal_array)
        sigma_array = f64(sigma_array)
        if np.any((sigma_array < 0) & np.isfinite(sigma_array)):
            raise ValueError("The standard deviation cannot be negative")
        ids = _allocate_ids(nominal_array.size).reshape(nominal_array.shape)
        gradient = np.ones(nominal_array.shape, dtype=np.float64)
        return cls(nominal_array, (_Term(ids, sigma_array, gradient),))

    @classmethod
    def constant(cls, value) -> "UArray":
        return cls(np.asarray(value, dtype=np.float64))

    @classmethod
    def from_scalar(cls, value: AffineScalarFunc) -> "UArray":
        nominal = np.asarray(value.nominal_value, dtype=np.float64)
        terms = []
        for variable, derivative in value.derivatives.items():
            identifier = _id_for_variable(variable)
            terms.append(
                _Term(
                    np.asarray(identifier, dtype=np.int64),
                    np.asarray(variable.std_dev, dtype=np.float64),
                    np.asarray(derivative, dtype=np.float64),
                )
            )
        return cls(nominal, tuple(terms))

    @property
    def nominal_values(self) -> np.ndarray:
        return self._nominal.copy()

    @property
    def std_devs(self) -> np.ndarray:
        variance = np.zeros(self.shape, dtype=np.float64)
        if not self.size:
            return variance
        kernel = lib()
        for term in self._terms:
            kernel.munc_variance_diag(
                addr(term.gradient), addr(term.sigma), addr(variance), self.size
            )
        diagonal = variance.copy()
        for index, left in enumerate(self._terms):
            for right in self._terms[index + 1 :]:
                kernel.munc_variance_cross(
                    addr(left.gradient),
                    addr(left.sigma),
                    addr(left.ids),
                    addr(right.gradient),
                    addr(right.sigma),
                    addr(right.ids),
                    addr(variance),
                    self.size,
                )
        kernel.munc_std_finish(addr(variance), addr(diagonal), self.size)
        return variance

    @property
    def shape(self) -> tuple[int, ...]:
        return self._nominal.shape

    @property
    def size(self) -> int:
        return self._nominal.size

    @property
    def ndim(self) -> int:
        return self._nominal.ndim

    @property
    def dtype(self):
        return np.dtype(object)

    @property
    def T(self) -> "UArray":
        return self.transpose()

    def __len__(self) -> int:
        return len(self._nominal)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"UArray(nominal_values={self._nominal!r}, std_devs={self.std_devs!r})"

    def _scalar_at(self, index) -> AffineScalarFunc:
        derivatives: dict[Variable, float] = {}
        for term in self._terms:
            identifier = int(term.ids[index])
            variable = _variable_for_id(identifier, float(term.sigma[index]))
            derivatives[variable] = derivatives.get(variable, 0.0) + float(term.gradient[index])
        return AffineScalarFunc(float(self._nominal[index]), derivatives)

    def __getitem__(self, index):
        nominal = self._nominal[index]
        if np.ndim(nominal) == 0:
            return self._scalar_at(index)
        terms = tuple(
            _Term(term.ids[index], term.sigma[index], term.gradient[index])
            for term in self._terms
        )
        return UArray(nominal, terms)

    def copy(self) -> "UArray":
        return UArray(
            self._nominal.copy(),
            tuple(
                _Term(term.ids.copy(), term.sigma.copy(), term.gradient.copy())
                for term in self._terms
            ),
        )

    def reshape(self, *shape, order="C") -> "UArray":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return UArray(
            self._nominal.reshape(*shape, order=order),
            tuple(
                _Term(
                    term.ids.reshape(*shape, order=order),
                    term.sigma.reshape(*shape, order=order),
                    term.gradient.reshape(*shape, order=order),
                )
                for term in self._terms
            ),
        )

    def ravel(self, order="C") -> "UArray":
        return self.reshape((-1,), order=order)

    def flatten(self, order="C") -> "UArray":
        return self.ravel(order=order).copy()

    def transpose(self, *axes) -> "UArray":
        axes_arg = axes if axes else None
        return UArray(
            self._nominal.transpose(axes_arg),
            tuple(
                _Term(
                    term.ids.transpose(axes_arg),
                    term.sigma.transpose(axes_arg),
                    term.gradient.transpose(axes_arg),
                )
                for term in self._terms
            ),
        )

    def to_object_array(self) -> np.ndarray:
        result = np.empty(self.shape, dtype=object)
        for index in np.ndindex(self.shape):
            result[index] = self._scalar_at(index)
        return result

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        result = self.to_object_array()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result.copy() if copy else result

    @staticmethod
    def _coerce(value) -> "UArray | None":
        if isinstance(value, UArray):
            return value
        if isinstance(value, AffineScalarFunc):
            return UArray.from_scalar(value)
        try:
            return UArray.constant(value)
        except (TypeError, ValueError):
            return None

    def _binary(self, other, operation: str, reflected: bool = False):
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        lhs = self
        if reflected:
            lhs, rhs = rhs, lhs
        left_nominal, right_nominal = np.broadcast_arrays(lhs._nominal, rhs._nominal)
        shape = left_nominal.shape
        left_nominal = f64(left_nominal)
        right_nominal = f64(right_nominal)
        values = np.empty(shape, dtype=np.float64)
        derivative_left = np.empty(shape, dtype=np.float64)
        derivative_right = np.empty(shape, dtype=np.float64)
        kernel = lib()
        if values.size:
            kernel.munc_binary(
                _BINARY_OPS[operation],
                addr(left_nominal),
                addr(right_nominal),
                addr(values),
                addr(derivative_left),
                addr(derivative_right),
                values.size,
            )
        terms = []
        left_terms = [_term_for_shape(term, shape) for term in lhs._terms]
        right_terms = [_term_for_shape(term, shape) for term in rhs._terms]
        used_right: set[int] = set()
        for current in left_terms:
            matching_index = next(
                (
                    index
                    for index, candidate in enumerate(right_terms)
                    if index not in used_right
                    and np.array_equal(current.ids, candidate.ids)
                    and np.array_equal(current.sigma, candidate.sigma)
                ),
                None,
            )
            gradient = np.empty(shape, dtype=np.float64)
            if not values.size:
                pass
            elif matching_index is None:
                kernel.munc_chain_unary(
                    addr(derivative_left),
                    addr(current.gradient),
                    addr(gradient),
                    values.size,
                )
            else:
                matching = right_terms[matching_index]
                used_right.add(matching_index)
                kernel.munc_chain_binary(
                    addr(derivative_left),
                    addr(derivative_right),
                    addr(current.gradient),
                    addr(matching.gradient),
                    addr(gradient),
                    values.size,
                )
            terms.append(_Term(current.ids, current.sigma, gradient))
        for index, current in enumerate(right_terms):
            if index in used_right:
                continue
            gradient = np.empty(shape, dtype=np.float64)
            if values.size:
                kernel.munc_chain_unary(
                    addr(derivative_right),
                    addr(current.gradient),
                    addr(gradient),
                    values.size,
                )
            terms.append(_Term(current.ids, current.sigma, gradient))
        return UArray(values, tuple(terms))

    def _unary(self, operation: str):
        values = np.empty(self.shape, dtype=np.float64)
        local = np.empty(self.shape, dtype=np.float64)
        kernel = lib()
        if self.size:
            kernel.munc_unary(
                _UNARY_OPS[operation],
                addr(self._nominal),
                addr(values),
                addr(local),
                self.size,
            )
        terms = []
        for term in self._terms:
            gradient = np.empty(self.shape, dtype=np.float64)
            if self.size:
                kernel.munc_chain_unary(
                    addr(local), addr(term.gradient), addr(gradient), self.size
                )
            terms.append(_Term(term.ids, term.sigma, gradient))
        return UArray(values, tuple(terms))

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self._binary(other, "add", True)

    def __sub__(self, other):
        return self._binary(other, "subtract")

    def __rsub__(self, other):
        return self._binary(other, "subtract", True)

    def __mul__(self, other):
        return self._binary(other, "multiply")

    def __rmul__(self, other):
        return self._binary(other, "multiply", True)

    def __truediv__(self, other):
        return self._binary(other, "divide")

    def __rtruediv__(self, other):
        return self._binary(other, "divide", True)

    def __pow__(self, other):
        return self._binary(other, "power")

    def __rpow__(self, other):
        return self._binary(other, "power", True)

    def __neg__(self):
        return self._unary("negative")

    def __pos__(self):
        return self._unary("positive")

    def __abs__(self):
        return self._unary("absolute")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        binary = {
            np.add: "add",
            np.subtract: "subtract",
            np.multiply: "multiply",
            np.divide: "divide",
            np.true_divide: "divide",
            np.power: "power",
        }
        unary = {
            np.negative: "negative",
            np.positive: "positive",
            np.absolute: "absolute",
            np.exp: "exp",
            np.expm1: "expm1",
            np.log: "log",
            np.log10: "log10",
            np.log1p: "log1p",
            np.sqrt: "sqrt",
            np.sin: "sin",
            np.cos: "cos",
            np.tan: "tan",
            np.arcsin: "asin",
            np.arccos: "acos",
            np.arctan: "atan",
            np.sinh: "sinh",
            np.cosh: "cosh",
            np.tanh: "tanh",
            np.arcsinh: "asinh",
            np.arccosh: "acosh",
            np.arctanh: "atanh",
            np.degrees: "degrees",
            np.radians: "radians",
        }
        if ufunc in binary:
            left = self._coerce(inputs[0])
            return left._binary(inputs[1], binary[ufunc])
        if ufunc in unary:
            return self._unary(unary[ufunc])
        return NotImplemented


def uarray(nominal_values, std_devs=None):
    if std_devs is None:
        raise TypeError("uarray() should be called with two arguments.")
    return UArray.independent(nominal_values, std_devs)


def nominal_values(arr):
    if isinstance(arr, UArray):
        return arr.nominal_values
    array = np.asanyarray(arr)
    return np.vectorize(nominal_value, otypes=[float])(array)


def std_devs(arr):
    if isinstance(arr, UArray):
        return arr.std_devs
    array = np.asanyarray(arr)
    return np.vectorize(std_dev, otypes=[float])(array)


class matrix(np.matrix):
    pass


def umatrix(nominal_values, std_devs=None):
    if std_devs is None:
        raise TypeError("umatrix() should be called with two arguments.")
    return matrix(uarray(nominal_values, std_devs).to_object_array())


__all__ = ["UArray", "uarray", "umatrix", "nominal_values", "std_devs", "matrix"]
