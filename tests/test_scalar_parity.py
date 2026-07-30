"""Scalar arithmetic is checked directly against uncertainties 3.x."""

import math

import numpy as np
import pytest

import uncertainties as upstream
from uncertainties import umath as upstream_math

import mojo_uncertainties as mojo
from mojo_uncertainties import umath as mojo_math


def assert_same(ours, theirs, *, rel=1e-12, abs=1e-14):
    if math.isnan(theirs.nominal_value):
        assert math.isnan(ours.nominal_value)
    else:
        assert ours.nominal_value == pytest.approx(theirs.nominal_value, rel=rel, abs=abs)
    assert ours.std_dev == pytest.approx(theirs.std_dev, rel=rel, abs=abs, nan_ok=True)


def pair(nominal=2.3, sigma=0.17):
    return mojo.ufloat(nominal, sigma), upstream.ufloat(nominal, sigma)


def test_constructor_properties_and_aliases():
    ours = mojo.ufloat(2.5, 0.3, tag="length")
    theirs = upstream.ufloat(2.5, 0.3, tag="length")
    assert ours.nominal_value == theirs.nominal_value
    assert ours.n == theirs.n
    assert ours.std_dev == theirs.std_dev
    assert ours.s == theirs.s
    assert ours.tag == theirs.tag
    assert isinstance(ours, mojo.UFloat)


def test_error_components_standard_score_and_basic_formatting():
    ours = mojo.ufloat(2.5, 0.3)
    theirs = upstream.ufloat(2.5, 0.3)
    assert list(ours.error_components().values()) == pytest.approx(
        list(theirs.error_components().values())
    )
    assert ours.std_score(3.1) == pytest.approx(theirs.std_score(3.1))
    assert format(ours, ".2f") == format(theirs, ".2f")


def test_mutating_source_uncertainty_updates_expression():
    ours = mojo.ufloat(2.0, 0.1)
    theirs = upstream.ufloat(2.0, 0.1)
    ours_expr = ours**2
    their_expr = theirs**2
    ours.std_dev = 0.4
    theirs.std_dev = 0.4
    assert_same(ours_expr, their_expr)


def test_negative_standard_deviation_rejected():
    with pytest.raises(mojo.NegativeStdDev):
        mojo.ufloat(1, -0.1)
    with pytest.raises(Exception) as upstream_error:
        upstream.ufloat(1, -0.1)
    assert upstream_error.type.__name__ == "NegativeStdDev"


@pytest.mark.parametrize(
    "operation",
    [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: y - x,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: y / x,
        lambda x, y: x**y,
        lambda x, y: 3.1**x,
    ],
)
def test_binary_arithmetic(operation):
    ours_x, theirs_x = pair()
    ours_y, theirs_y = pair(1.4, 0.08)
    assert_same(operation(ours_x, ours_y), operation(theirs_x, theirs_y))


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: +x,
        lambda x: -x,
        lambda x: abs(x),
        lambda x: 3 + x,
        lambda x: 3 - x,
        lambda x: 3 * x,
        lambda x: 3 / x,
    ],
)
def test_unary_and_constant_arithmetic(operation):
    ours, theirs = pair(-2.3, 0.17)
    assert_same(operation(ours), operation(theirs))


def test_exact_correlation_cancellation():
    ours, theirs = pair()
    assert_same(ours - ours, theirs - theirs)
    assert (ours - ours).std_dev == 0.0
    assert_same(ours / ours, theirs / theirs)


def test_shared_variable_covariance_in_expression():
    ours_x, theirs_x = pair(2.0, 0.2)
    ours_y, theirs_y = pair(3.0, 0.4)
    ours = (ours_x + ours_y) * (ours_x - ours_y)
    theirs = (theirs_x + theirs_y) * (theirs_x - theirs_y)
    assert_same(ours, theirs)
    assert ours.derivatives[ours_x] == pytest.approx(theirs.derivatives[theirs_x])
    assert ours.derivatives[ours_y] == pytest.approx(theirs.derivatives[theirs_y])


def test_self_multiplication_derivative_parity():
    ours, theirs = pair(1.2, 0.02)
    ours_result = ours * ours
    theirs_result = theirs * theirs
    assert_same(ours_result, theirs_result)
    assert ours_result.derivatives[ours] == pytest.approx(
        theirs_result.derivatives[theirs]
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("sin", 0.4),
        ("cos", 0.4),
        ("tan", 0.4),
        ("asin", 0.4),
        ("acos", 0.4),
        ("atan", 0.4),
        ("sinh", 0.4),
        ("cosh", 0.4),
        ("tanh", 0.4),
        ("asinh", 0.4),
        ("acosh", 1.4),
        ("atanh", 0.4),
        ("exp", 0.4),
        ("expm1", 0.4),
        ("log", 1.4),
        ("log10", 1.4),
        ("log1p", 0.4),
        ("sqrt", 1.4),
        ("erf", 0.4),
        ("erfc", 0.4),
        ("degrees", 0.4),
        ("radians", 0.4),
    ],
)
def test_umath_unary(name, value):
    ours, theirs = pair(value, 0.03)
    assert_same(getattr(mojo_math, name)(ours), getattr(upstream_math, name)(theirs))


@pytest.mark.parametrize("name", ["atan2", "hypot", "copysign"])
def test_umath_binary(name):
    ours_x, theirs_x = pair(1.2, 0.04)
    ours_y, theirs_y = pair(2.4, 0.07)
    assert_same(
        getattr(mojo_math, name)(ours_x, ours_y),
        getattr(upstream_math, name)(theirs_x, theirs_y),
    )


def test_remaining_umath_arithmetic_helpers():
    ours_x, theirs_x = pair(2.4, 0.03)
    ours_y, theirs_y = pair(1.3, 0.02)
    assert_same(mojo_math.fmod(ours_x, ours_y), upstream_math.fmod(theirs_x, theirs_y), rel=2e-8)
    for name in ("gamma", "lgamma"):
        assert_same(
            getattr(mojo_math, name)(ours_x),
            getattr(upstream_math, name)(theirs_x),
            rel=2e-8,
        )
    ours_fraction, ours_integer = mojo_math.modf(ours_x)
    their_fraction, their_integer = upstream_math.modf(theirs_x)
    assert_same(ours_fraction, their_fraction)
    assert ours_integer == their_integer
    ours_mantissa, ours_exponent = mojo_math.frexp(ours_x)
    their_mantissa, their_exponent = upstream_math.frexp(theirs_x)
    assert ours_mantissa.nominal_value == their_mantissa.nominal_value
    assert ours_mantissa.std_dev == pytest.approx(0.03 * 2**-ours_exponent)
    assert ours_exponent == their_exponent
    assert_same(mojo_math.ldexp(ours_x, 3), upstream_math.ldexp(theirs_x, 3))


def test_log_with_uncertain_base():
    ours_x, theirs_x = pair(8.0, 0.2)
    ours_b, theirs_b = pair(2.0, 0.03)
    assert_same(mojo_math.log(ours_x, ours_b), upstream_math.log(theirs_x, theirs_b))


def test_fsum_preserves_correlation():
    ours, theirs = pair()
    assert_same(
        mojo_math.fsum([ours, 2 * ours, -ours]),
        upstream_math.fsum([theirs, 2 * theirs, -theirs]),
    )


def test_wrap_with_analytic_derivatives():
    def function(x, y=1.0):
        return math.sin(x) * y

    ours = mojo.wrap(
        function,
        derivatives_args=[lambda x, y=1.0: math.cos(x) * y],
        derivatives_kwargs={"y": lambda x, y=1.0: math.sin(x)},
    )
    theirs = upstream.wrap(
        function,
        derivatives_args=[lambda x, y=1.0: math.cos(x) * y],
        derivatives_kwargs={"y": lambda x, y=1.0: math.sin(x)},
    )
    ours_x, theirs_x = pair(0.7, 0.02)
    ours_y, theirs_y = pair(1.4, 0.05)
    assert_same(ours(ours_x, y=ours_y), theirs(theirs_x, y=theirs_y), rel=1e-8)


def test_wrap_with_numerical_derivatives():
    ours = mojo.wrap(lambda x: math.gamma(x))
    theirs = upstream.wrap(lambda x: math.gamma(x))
    ours_x, theirs_x = pair(2.4, 0.03)
    assert_same(ours(ours_x), theirs(theirs_x), rel=2e-8)


@pytest.mark.parametrize(
    "text",
    [
        "12.58+/-0.23",
        "12.58 ± 0.23",
        "3.85e5 +/- 2.3e4",
        "(38.5 +/- 2.3)e4",
        "72.1(2.2)",
        "72.15(4)",
        "680(41)e-3",
        "23.29",
        "nan",
        "680.3(nan)",
    ],
)
def test_ufloat_fromstr(text):
    assert_same(mojo.ufloat_fromstr(text), upstream.ufloat_fromstr(text))


def test_covariance_and_correlation_matrices():
    ours_x = mojo.ufloat(1.0, 0.1)
    ours_y = mojo.ufloat(2.0, 0.3)
    theirs_x = upstream.ufloat(1.0, 0.1)
    theirs_y = upstream.ufloat(2.0, 0.3)
    ours_values = [ours_x, ours_y, ours_x + 2 * ours_y]
    their_values = [theirs_x, theirs_y, theirs_x + 2 * theirs_y]
    assert np.allclose(
        mojo.covariance_matrix(ours_values),
        upstream.covariance_matrix(their_values),
    )
    assert np.allclose(
        mojo.correlation_matrix(ours_values),
        upstream.correlation_matrix(their_values),
    )


def test_correlated_values_reproduce_covariance():
    nominal = [1.0, 2.0, -1.0]
    covariance = np.array(
        [[0.04, 0.012, -0.003], [0.012, 0.09, 0.006], [-0.003, 0.006, 0.01]]
    )
    ours = mojo.correlated_values(nominal, covariance, tags=["a", "b", "c"])
    theirs = upstream.correlated_values(nominal, covariance, tags=["a", "b", "c"])
    assert np.allclose(mojo.covariance_matrix(ours), covariance)
    assert np.allclose(
        mojo.covariance_matrix(ours),
        upstream.covariance_matrix(theirs),
    )
    assert_same(ours[0] + 2 * ours[1] - ours[2], theirs[0] + 2 * theirs[1] - theirs[2])


def test_correlated_values_norm_zero_variance():
    values = [(1.0, 0.2), (2.0, 0.0)]
    correlation = [[1.0, 0.0], [0.0, 1.0]]
    ours = mojo.correlated_values_norm(values, correlation)
    theirs = upstream.correlated_values_norm(values, correlation)
    assert_same(ours[0], theirs[0])
    assert_same(ours[1], theirs[1])


def test_uniform_access_for_plain_numbers():
    assert mojo.nominal_value(3.0) == upstream.nominal_value(3.0)
    assert mojo.std_dev(3.0) == upstream.std_dev(3.0)


def test_comparisons_match_upstream():
    ours, theirs = pair()
    assert (ours == ours) == (theirs == theirs)
    assert (ours == ours.nominal_value) == (theirs == theirs.nominal_value)
    assert (ours > 1.0) == (theirs > 1.0)
    assert (ours < 3.0) == (theirs < 3.0)
