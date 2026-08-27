"""Scalar affine error propagation compatible with the covered upstream API."""

from __future__ import annotations

import math
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np


class NegativeStdDev(Exception):
    """Raised when a variable is assigned a negative standard deviation."""


class AffineScalarFunc:
    """A scalar value represented by its nominal value and first derivatives."""

    __slots__ = ("_nominal_value", "_derivatives")
    __array_priority__ = 1000

    class dtype:
        type = staticmethod(lambda value: value)

    def __init__(self, nominal_value: float, derivatives: Mapping["Variable", float]):
        self._nominal_value = float(nominal_value)
        self._derivatives = dict(derivatives)

    @staticmethod
    def _from_owned_derivatives(
        nominal_value: float, derivatives: dict["Variable", float]
    ) -> "AffineScalarFunc":
        result = object.__new__(AffineScalarFunc)
        result._nominal_value = float(nominal_value)
        result._derivatives = derivatives
        return result

    @property
    def nominal_value(self) -> float:
        return self._nominal_value

    n = nominal_value

    @property
    def derivatives(self) -> dict["Variable", float]:
        return self._derivatives

    def error_components(self) -> dict["Variable", float]:
        return {
            variable: 0.0
            if variable.std_dev == 0
            else abs(derivative * variable.std_dev)
            for variable, derivative in self.derivatives.items()
        }

    @property
    def std_dev(self) -> float:
        return float(math.sqrt(math.fsum(x * x for x in self.error_components().values())))

    s = std_dev

    def std_score(self, value: float) -> float:
        if self.std_dev == 0:
            raise ValueError("The standard deviation is zero: undefined result")
        return (value - self.nominal_value) / self.std_dev

    def __repr__(self) -> str:
        sigma = repr(self.std_dev) if self.std_dev else "0"
        return f"{self.nominal_value!r}+/-{sigma}"

    def __str__(self) -> str:
        return f"{self.nominal_value}+/-{self.std_dev}"

    def __format__(self, format_spec: str) -> str:
        if not format_spec:
            return str(self)
        return f"{format(self.nominal_value, format_spec)}+/-{format(self.std_dev, format_spec)}"

    def format(self, format_spec: str) -> str:
        return format(self, format_spec)

    @staticmethod
    def _coerce(value: Any) -> "AffineScalarFunc | None":
        if isinstance(value, AffineScalarFunc):
            return value
        if isinstance(value, (int, float, np.number)):
            return AffineScalarFunc(float(value), {})
        return None

    @staticmethod
    def _combine(
        left: Mapping["Variable", float],
        left_scale: float,
        right: Mapping["Variable", float],
        right_scale: float,
    ) -> dict["Variable", float]:
        result = {variable: left_scale * value for variable, value in left.items()}
        for variable, value in right.items():
            result[variable] = result.get(variable, 0.0) + right_scale * value
        return result

    def _binary(
        self,
        other: Any,
        value: Callable[[float, float], float],
        derivative_left: Callable[[float, float], float],
        derivative_right: Callable[[float, float], float],
    ):
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        x, y = self.nominal_value, rhs.nominal_value
        nominal = value(x, y)
        return self._from_owned_derivatives(
            nominal,
            self._combine(
                self.derivatives,
                derivative_left(x, y),
                rhs.derivatives,
                derivative_right(x, y),
            ),
        )

    def __add__(self, other: Any):
        if isinstance(other, AffineScalarFunc):
            nominal = other._nominal_value
            other_derivatives = other._derivatives
        elif isinstance(other, (int, float, np.number)):
            result = object.__new__(AffineScalarFunc)
            result._nominal_value = self._nominal_value + float(other)
            result._derivatives = self._derivatives.copy()
            return result
        else:
            return NotImplemented
        derivatives = self._derivatives.copy()
        for variable in other_derivatives:
            derivatives[variable] = (
                derivatives.get(variable, 0.0) + other_derivatives[variable]
            )
        result = object.__new__(AffineScalarFunc)
        result._nominal_value = self._nominal_value + nominal
        result._derivatives = derivatives
        return result

    __radd__ = __add__

    def __sub__(self, other: Any):
        if isinstance(other, AffineScalarFunc):
            nominal = other._nominal_value
            other_derivatives = other._derivatives
        elif isinstance(other, (int, float, np.number)):
            result = object.__new__(AffineScalarFunc)
            result._nominal_value = self._nominal_value - float(other)
            result._derivatives = self._derivatives.copy()
            return result
        else:
            return NotImplemented
        derivatives = self._derivatives.copy()
        for variable in other_derivatives:
            derivatives[variable] = (
                derivatives.get(variable, 0.0) - other_derivatives[variable]
            )
        result = object.__new__(AffineScalarFunc)
        result._nominal_value = self._nominal_value - nominal
        result._derivatives = derivatives
        return result

    def __rsub__(self, other: Any):
        lhs = self._coerce(other)
        return NotImplemented if lhs is None else lhs.__sub__(self)

    def __mul__(self, other: Any):
        if other is self:
            derivatives = {
                variable: 2.0 * self._nominal_value * derivative
                for variable, derivative in self._derivatives.items()
            }
            result = object.__new__(AffineScalarFunc)
            result._nominal_value = self._nominal_value * self._nominal_value
            result._derivatives = derivatives
            return result
        if isinstance(other, AffineScalarFunc):
            nominal = other._nominal_value
            other_derivatives = other._derivatives
        elif isinstance(other, (int, float, np.number)):
            nominal = float(other)
            other_derivatives = ()
        else:
            return NotImplemented
        derivatives = {
            variable: nominal * derivative
            for variable, derivative in self._derivatives.items()
        }
        for variable, derivative in (
            other_derivatives.items() if other_derivatives else ()
        ):
            derivatives[variable] = (
                derivatives.get(variable, 0.0) + self._nominal_value * derivative
            )
        result = object.__new__(AffineScalarFunc)
        result._nominal_value = self._nominal_value * nominal
        result._derivatives = derivatives
        return result

    __rmul__ = __mul__

    def __truediv__(self, other: Any):
        return self._binary(
            other,
            lambda x, y: x / y,
            lambda x, y: 1.0 / y,
            lambda x, y: -x / (y * y),
        )

    def __rtruediv__(self, other: Any):
        lhs = self._coerce(other)
        return NotImplemented if lhs is None else lhs.__truediv__(self)

    @staticmethod
    def _pow_dx(x: float, y: float) -> float:
        if x > 0 or (y % 1 == 0 and (x < 0 or y >= 1)):
            return y * x ** (y - 1)
        if x == 0 and y == 0:
            return 0.0
        return math.nan

    @staticmethod
    def _pow_dy(x: float, y: float) -> float:
        if x > 0:
            return math.log(x) * x**y
        if x == 0 and y > 0:
            return 0.0
        return math.nan

    def __pow__(self, other: Any):
        return self._binary(other, lambda x, y: x**y, self._pow_dx, self._pow_dy)

    def __rpow__(self, other: Any):
        lhs = self._coerce(other)
        return NotImplemented if lhs is None else lhs.__pow__(self)

    def __neg__(self):
        return AffineScalarFunc(-self.nominal_value, {v: -d for v, d in self.derivatives.items()})

    def __pos__(self):
        return AffineScalarFunc(self.nominal_value, self.derivatives)

    def __abs__(self):
        derivative = 1.0 if self.nominal_value >= 0 else -1.0
        return AffineScalarFunc(
            abs(self.nominal_value), {v: derivative * d for v, d in self.derivatives.items()}
        )

    def __mod__(self, other: Any):
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        x, y = self.nominal_value, rhs.nominal_value
        step = math.sqrt(sys.float_info.epsilon) * max(abs(y), 1.0)
        dy = ((x % (y + step)) - (x % (y - step))) / (2 * step)
        return AffineScalarFunc(
            x % y, self._combine(self.derivatives, 1.0, rhs.derivatives, dy)
        )

    def __rmod__(self, other: Any):
        lhs = self._coerce(other)
        return NotImplemented if lhs is None else lhs.__mod__(self)

    def __floordiv__(self, other: Any):
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return AffineScalarFunc(self.nominal_value // rhs.nominal_value, {})

    def __rfloordiv__(self, other: Any):
        lhs = self._coerce(other)
        return NotImplemented if lhs is None else lhs.__floordiv__(self)

    def __eq__(self, other: Any) -> bool:
        rhs = self._coerce(other)
        if rhs is None:
            return False
        difference = self - rhs
        return not (difference.nominal_value or difference.std_dev)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __lt__(self, other: Any) -> bool:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return self.nominal_value < rhs.nominal_value

    def __le__(self, other: Any) -> bool:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return self < rhs or self == rhs

    def __gt__(self, other: Any) -> bool:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return self.nominal_value > rhs.nominal_value

    def __ge__(self, other: Any) -> bool:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return self > rhs or self == rhs

    def __bool__(self) -> bool:
        return bool(self.nominal_value)

    def __float__(self):
        raise TypeError("can't convert an affine function to float; use x.nominal_value")

    __int__ = __float__
    __complex__ = __float__


UFloat = AffineScalarFunc


class Variable(AffineScalarFunc):
    """An independent uncertain scalar."""

    __slots__ = ("_std_dev", "tag")

    def __init__(self, value: float, std_dev: float, tag: Any = None):
        self._std_dev = 0.0
        self.tag = tag
        super().__init__(value, {self: 1.0})
        self.std_dev = std_dev

    __hash__ = object.__hash__

    @property
    def std_dev(self) -> float:
        return self._std_dev

    @std_dev.setter
    def std_dev(self, value: float) -> None:
        if value < 0 and math.isfinite(value):
            raise NegativeStdDev("The standard deviation cannot be negative")
        self._std_dev = float(value)

    s = std_dev

    def __repr__(self) -> str:
        basic = super().__repr__()
        return basic if self.tag is None else f"< {self.tag} = {basic} >"


def ufloat(nominal_value, std_dev=None, tag=None) -> Variable:
    if std_dev == 0:
        warnings.warn("Using UFloat objects with std_dev==0 may give unexpected results.")
    return Variable(nominal_value, std_dev, tag=tag)


_PLUS_MINUS = re.compile(
    r"^\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf)(?:e[+-]?\d+)?)"
    r"\s*(?:\+/-|±)\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf)(?:e[+-]?\d+)?)\s*\)?"
    r"(?:e([+-]?\d+))?$",
    re.IGNORECASE,
)
_PAREN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf))"
    r"(?:\(([^)]+)\))?(?:e([+-]?\d+))?$",
    re.IGNORECASE,
)


def ufloat_fromstr(representation, tag=None) -> Variable:
    text = representation.strip()
    match = _PLUS_MINUS.match(text)
    if match:
        exponent = int(match.group(3) or 0)
        scale = 10.0**exponent
        return ufloat(float(match.group(1)) * scale, float(match.group(2)) * scale, tag)

    match = _PAREN.match(text)
    if not match:
        raise ValueError(f"Cannot parse {representation!r}")
    nominal_text, uncertainty_text, exponent_text = match.groups()
    exponent = int(exponent_text or 0)
    nominal = float(nominal_text) * 10.0**exponent
    if uncertainty_text is None:
        if not math.isfinite(float(nominal_text)):
            uncertainty = 1.0
        else:
            decimals = len(nominal_text.partition(".")[2])
            uncertainty = 10.0 ** (exponent - decimals)
    elif uncertainty_text.lower() == "nan" or "." in uncertainty_text:
        uncertainty = float(uncertainty_text) * 10.0**exponent
    else:
        decimals = len(nominal_text.partition(".")[2])
        uncertainty = float(uncertainty_text) * 10.0 ** (exponent - decimals)
    return ufloat(nominal, uncertainty, tag)


def nominal_value(x):
    return x.nominal_value if isinstance(x, AffineScalarFunc) else x


def std_dev(x) -> float:
    return x.std_dev if isinstance(x, AffineScalarFunc) else 0.0


def covariance_matrix(nums_with_uncert: Sequence[AffineScalarFunc]) -> list[list[float]]:
    values = list(nums_with_uncert)
    matrix = [[0.0] * len(values) for _ in values]
    for i, left in enumerate(values):
        for j in range(i + 1):
            right = values[j]
            shared = left.derivatives.keys() & right.derivatives.keys()
            covariance = math.fsum(
                left.derivatives[var] * right.derivatives[var] * var.std_dev**2
                for var in shared
            )
            matrix[i][j] = matrix[j][i] = float(covariance)
    return matrix


def correlation_matrix(nums_with_uncert: Sequence[AffineScalarFunc]) -> np.ndarray:
    covariance = np.asarray(covariance_matrix(nums_with_uncert), dtype=np.float64)
    sigma = np.sqrt(np.diag(covariance))
    return covariance / sigma / sigma[:, np.newaxis]


def correlated_values(nom_values, covariance_mat, tags=None):
    covariance = np.asarray(covariance_mat, dtype=np.float64)
    sigma = np.sqrt(np.diag(covariance))
    normalizer = sigma.copy()
    normalizer[normalizer == 0] = 1.0
    correlation = covariance / normalizer / normalizer[:, np.newaxis]
    return correlated_values_norm(list(zip(nom_values, sigma)), correlation, tags)


def correlated_values_norm(values_with_std_dev, correlation_mat, tags=None):
    pairs = np.asarray(values_with_std_dev, dtype=np.float64)
    nominal = pairs[:, 0]
    sigma = pairs[:, 1]
    if tags is None:
        tags = (None,) * len(pairs)
    variances, transform = np.linalg.eigh(np.asarray(correlation_mat, dtype=np.float64))
    variances[variances < 0] = 0.0
    variables = tuple(
        Variable(0.0, math.sqrt(variance), tag)
        for variance, tag in zip(variances, tags)
    )
    transform *= sigma[:, np.newaxis]
    return tuple(
        AffineScalarFunc(value, dict(zip(variables, coordinates)))
        for value, coordinates in zip(nominal, transform)
    )


def _numerical_derivative(function: Callable, argument: int | str):
    def derivative(*args, **kwargs):
        positional = list(args)
        if isinstance(argument, str):
            value = kwargs[argument]
        else:
            value = positional[argument]
        step = math.sqrt(sys.float_info.epsilon) * abs(value)
        if not step:
            step = math.sqrt(sys.float_info.epsilon)
        if isinstance(argument, str):
            plus = dict(kwargs)
            minus = dict(kwargs)
            plus[argument] = value + step
            minus[argument] = value - step
            return (function(*args, **plus) - function(*args, **minus)) / (2 * step)
        positional[argument] = value + step
        plus_value = function(*positional, **kwargs)
        positional[argument] = value - step
        return (plus_value - function(*positional, **kwargs)) / (2 * step)

    return derivative


def wrap(f, derivatives_args=None, derivatives_kwargs=None):
    positional_derivatives = list(derivatives_args or ())
    keyword_derivatives = dict(derivatives_kwargs or {})

    def wrapped(*args, **kwargs):
        nominal_args = [nominal_value(value) for value in args]
        nominal_kwargs = {name: nominal_value(value) for name, value in kwargs.items()}
        result = f(*nominal_args, **nominal_kwargs)
        uncertain_args = [
            (index, value) for index, value in enumerate(args) if isinstance(value, AffineScalarFunc)
        ]
        uncertain_kwargs = [
            (name, value) for name, value in kwargs.items() if isinstance(value, AffineScalarFunc)
        ]
        if not uncertain_args and not uncertain_kwargs:
            return result
        derivatives: dict[Variable, float] = {}
        for index, value in uncertain_args:
            derivative = (
                positional_derivatives[index]
                if index < len(positional_derivatives) and positional_derivatives[index] is not None
                else _numerical_derivative(f, index)
            )
            local = derivative(*nominal_args, **nominal_kwargs)
            for variable, coefficient in value.derivatives.items():
                derivatives[variable] = derivatives.get(variable, 0.0) + local * coefficient
        for name, value in uncertain_kwargs:
            derivative = keyword_derivatives.get(name) or _numerical_derivative(f, name)
            local = derivative(*nominal_args, **nominal_kwargs)
            for variable, coefficient in value.derivatives.items():
                derivatives[variable] = derivatives.get(variable, 0.0) + local * coefficient
        return AffineScalarFunc(result, derivatives)

    wrapped.__name__ = getattr(f, "__name__", "wrapped")
    wrapped.__doc__ = getattr(f, "__doc__")
    return wrapped


def nan_if_exception(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan

    return wrapped


modified_operators = ["abs", "neg", "pos", "trunc"]
modified_ops_with_reflection = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
]


__all__ = [
    "ufloat",
    "ufloat_fromstr",
    "nominal_value",
    "std_dev",
    "covariance_matrix",
    "correlation_matrix",
    "correlated_values",
    "correlated_values_norm",
    "UFloat",
    "Variable",
    "AffineScalarFunc",
    "NegativeStdDev",
    "wrap",
    "nan_if_exception",
    "modified_operators",
    "modified_ops_with_reflection",
]
