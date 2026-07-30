"""Array helpers matching the arithmetic subset of :mod:`uncertainties.unumpy`."""

from __future__ import annotations

import numpy as np

from .. import umath
from .core import UArray, matrix, nominal_values, std_devs, uarray, umatrix


def _unary(name):
    scalar_function = getattr(umath, name)

    def function(value):
        if isinstance(value, UArray):
            return value._unary(name)
        array = np.asanyarray(value)
        return np.vectorize(scalar_function, otypes=[object])(array)

    function.__name__ = name
    return function


for _name in [
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "degrees",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "log",
    "log10",
    "log1p",
    "radians",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]:
    globals()[_name] = _unary(_name)

arccos = acos
arccosh = acosh
arcsin = asin
arctan = atan
arctanh = atanh


def pow(left, right):
    if isinstance(left, UArray):
        return left**right
    if isinstance(right, UArray):
        return right.__rpow__(left)
    return np.vectorize(umath.pow, otypes=[object])(left, right)


def _binary(name, left, right):
    if isinstance(left, UArray):
        return left._binary(right, name)
    if isinstance(right, UArray):
        return right._binary(left, name, reflected=True)
    return np.vectorize(getattr(umath, name), otypes=[object])(left, right)


def arctan2(y, x):
    return _binary("atan2", y, x)


def hypot(x, y):
    return _binary("hypot", x, y)


def copysign(x, y):
    return _binary("copysign", x, y)


def fmod(x, y):
    return _binary("fmod", x, y)


def absolute(value):
    return abs(value) if isinstance(value, UArray) else np.vectorize(abs, otypes=[object])(value)


fabs = absolute


def _locally_constant(name, value):
    nominal = nominal_values(value)
    return getattr(np, name)(nominal)


def ceil(value):
    return _locally_constant("ceil", value)


def floor(value):
    return _locally_constant("floor", value)


def trunc(value):
    return _locally_constant("trunc", value)


def isinf(value):
    return _locally_constant("isinf", value)


def isnan(value):
    return _locally_constant("isnan", value)


def modf(value):
    if isinstance(value, UArray):
        integer = np.trunc(value.nominal_values)
        fraction = (value - integer).to_object_array()
        result = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            result[index] = (fraction[index], float(integer[index]))
        return result
    return np.vectorize(umath.modf, otypes=[object, float])(value)


def ldexp(value, exponent):
    if isinstance(value, UArray):
        return value * np.exp2(exponent)
    return np.vectorize(umath.ldexp, otypes=[object])(value, exponent)


def gamma(value):
    source = value.to_object_array() if isinstance(value, UArray) else value
    return np.vectorize(umath.gamma, otypes=[object])(source)


def lgamma(value):
    source = value.to_object_array() if isinstance(value, UArray) else value
    return np.vectorize(umath.lgamma, otypes=[object])(source)

__all__ = [
    "UArray",
    "uarray",
    "umatrix",
    "nominal_values",
    "std_devs",
    "matrix",
    "absolute",
    "arctan2",
    "ceil",
    "copysign",
    "fabs",
    "floor",
    "fmod",
    "gamma",
    "hypot",
    "isinf",
    "isnan",
    "ldexp",
    "lgamma",
    "modf",
    "pow",
    "trunc",
    "arccos",
    "arccosh",
    "arcsin",
    "asinh",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "degrees",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "log",
    "log10",
    "log1p",
    "radians",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]
