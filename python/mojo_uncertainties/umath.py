"""Uncertainty-aware counterparts of the standard :mod:`math` functions."""

from __future__ import annotations

import math
from typing import Callable

from .core import AffineScalarFunc, wrap


def _array_unary(value, name: str):
    from .unumpy import UArray

    return value._unary(name) if isinstance(value, UArray) else None


def _unary(value, function: Callable[[float], float], derivative: Callable[[float], float], name: str):
    array_result = _array_unary(value, name)
    if array_result is not None:
        return array_result
    if not isinstance(value, AffineScalarFunc):
        return function(value)
    x = value.nominal_value
    local = derivative(x)
    return AffineScalarFunc(function(x), {var: local * grad for var, grad in value.derivatives.items()})


def _binary(
    left,
    right,
    function: Callable[[float, float], float],
    derivative_left: Callable[[float, float], float],
    derivative_right: Callable[[float, float], float],
):
    if not isinstance(left, AffineScalarFunc) and not isinstance(right, AffineScalarFunc):
        return function(left, right)
    lhs = AffineScalarFunc._coerce(left)
    rhs = AffineScalarFunc._coerce(right)
    if lhs is None or rhs is None:
        return NotImplemented
    x, y = lhs.nominal_value, rhs.nominal_value
    return AffineScalarFunc(
        function(x, y),
        AffineScalarFunc._combine(
            lhs.derivatives,
            derivative_left(x, y),
            rhs.derivatives,
            derivative_right(x, y),
        ),
    )


def sin(x):
    return _unary(x, math.sin, math.cos, "sin")


def cos(x):
    return _unary(x, math.cos, lambda v: -math.sin(v), "cos")


def tan(x):
    return _unary(x, math.tan, lambda v: 1 + math.tan(v) ** 2, "tan")


def asin(x):
    return _unary(x, math.asin, lambda v: 1 / math.sqrt(1 - v * v), "asin")


def acos(x):
    return _unary(x, math.acos, lambda v: -1 / math.sqrt(1 - v * v), "acos")


def atan(x):
    return _unary(x, math.atan, lambda v: 1 / (1 + v * v), "atan")


def sinh(x):
    return _unary(x, math.sinh, math.cosh, "sinh")


def cosh(x):
    return _unary(x, math.cosh, math.sinh, "cosh")


def tanh(x):
    return _unary(x, math.tanh, lambda v: 1 - math.tanh(v) ** 2, "tanh")


def asinh(x):
    return _unary(x, math.asinh, lambda v: 1 / math.sqrt(1 + v * v), "asinh")


def acosh(x):
    return _unary(x, math.acosh, lambda v: 1 / math.sqrt(v * v - 1), "acosh")


def atanh(x):
    return _unary(x, math.atanh, lambda v: 1 / (1 - v * v), "atanh")


def exp(x):
    return _unary(x, math.exp, math.exp, "exp")


def expm1(x):
    return _unary(x, math.expm1, math.exp, "expm1")


def log(x, base=None):
    if base is None:
        return _unary(x, math.log, lambda v: 1 / v, "log")
    return _binary(
        x,
        base,
        math.log,
        lambda value, radix: 1 / value / math.log(radix),
        lambda value, radix: -math.log(value, radix) / radix / math.log(radix),
    )


def log10(x):
    return _unary(x, math.log10, lambda v: 1 / v / math.log(10), "log10")


def log1p(x):
    return _unary(x, math.log1p, lambda v: 1 / (1 + v), "log1p")


def sqrt(x):
    return _unary(x, math.sqrt, lambda v: 0.5 / math.sqrt(v), "sqrt")


def erf(x):
    return _unary(
        x,
        math.erf,
        lambda v: math.exp(-(v * v)) * 2 / math.sqrt(math.pi),
        "erf",
    )


def erfc(x):
    return _unary(
        x,
        math.erfc,
        lambda v: -math.exp(-(v * v)) * 2 / math.sqrt(math.pi),
        "erfc",
    )


def degrees(x):
    return _unary(x, math.degrees, lambda v: math.degrees(1), "degrees")


def radians(x):
    return _unary(x, math.radians, lambda v: math.radians(1), "radians")


def fabs(x):
    return abs(x) if isinstance(x, AffineScalarFunc) else math.fabs(x)


def pow(x, y):
    return x**y if isinstance(x, AffineScalarFunc) or isinstance(y, AffineScalarFunc) else math.pow(x, y)


def atan2(y, x):
    return _binary(
        y,
        x,
        math.atan2,
        lambda first, second: second / (first * first + second * second),
        lambda first, second: -first / (first * first + second * second),
    )


def hypot(x, y):
    return _binary(
        x,
        y,
        math.hypot,
        lambda first, second: first / math.hypot(first, second),
        lambda first, second: second / math.hypot(first, second),
    )


def copysign(x, y):
    return _binary(
        x,
        y,
        math.copysign,
        lambda first, second: math.copysign(1, second) if first >= 0 else -math.copysign(1, second),
        lambda first, second: 0.0,
    )


def fmod(x, y):
    return wrap(math.fmod)(x, y)


def gamma(x):
    return wrap(math.gamma)(x)


def lgamma(x):
    return wrap(math.lgamma)(x)


def ceil(x):
    return math.ceil(x.nominal_value if isinstance(x, AffineScalarFunc) else x)


def floor(x):
    return math.floor(x.nominal_value if isinstance(x, AffineScalarFunc) else x)


def trunc(x):
    return math.trunc(x.nominal_value if isinstance(x, AffineScalarFunc) else x)


def isinf(x):
    return math.isinf(x.nominal_value if isinstance(x, AffineScalarFunc) else x)


def isnan(x):
    return math.isnan(x.nominal_value if isinstance(x, AffineScalarFunc) else x)


def fsum(values):
    values = list(values)
    result = 0.0
    for value in values:
        result = result + value
    return result


def modf(x):
    if not isinstance(x, AffineScalarFunc):
        return math.modf(x)
    fraction, integer = math.modf(x.nominal_value)
    return AffineScalarFunc(fraction, x.derivatives), integer


def ldexp(x, i):
    if not isinstance(x, AffineScalarFunc):
        return math.ldexp(x, i)
    scale = 2.0**i
    return AffineScalarFunc(
        math.ldexp(x.nominal_value, i),
        {variable: scale * derivative for variable, derivative in x.derivatives.items()},
    )


def frexp(x):
    if not isinstance(x, AffineScalarFunc):
        return math.frexp(x)
    mantissa, exponent = math.frexp(x.nominal_value)
    scale = 2.0**-exponent
    return (
        AffineScalarFunc(
            mantissa,
            {variable: scale * derivative for variable, derivative in x.derivatives.items()},
        ),
        exponent,
    )


factorial = math.factorial

__all__ = [
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "copysign",
    "cos",
    "cosh",
    "degrees",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "fmod",
    "frexp",
    "fsum",
    "gamma",
    "hypot",
    "isinf",
    "isnan",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "modf",
    "pow",
    "radians",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]
