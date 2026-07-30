"""Mojo array kernels are checked against upstream unumpy object arrays."""

import numpy as np
import pytest

from uncertainties import unumpy as upstream

from mojo_uncertainties import ufloat
from mojo_uncertainties import unumpy as mojo


def assert_array_same(ours, theirs, *, rtol=1e-12, atol=1e-14):
    assert np.allclose(
        mojo.nominal_values(ours),
        upstream.nominal_values(theirs),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    assert np.allclose(
        mojo.std_devs(ours),
        upstream.std_devs(theirs),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


@pytest.fixture
def arrays():
    nominal_a = np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    sigma_a = np.array([[0.02, 0.03, 0.04], [0.05, 0.06, 0.07]])
    nominal_b = np.array([[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]])
    sigma_b = np.array([[0.08, 0.07, 0.06], [0.05, 0.04, 0.03]])
    return (
        mojo.uarray(nominal_a, sigma_a),
        mojo.uarray(nominal_b, sigma_b),
        upstream.uarray(nominal_a, sigma_a),
        upstream.uarray(nominal_b, sigma_b),
    )


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
    ],
)
def test_binary_array_arithmetic(arrays, operation):
    ours_a, ours_b, theirs_a, theirs_b = arrays
    assert_array_same(operation(ours_a, ours_b), operation(theirs_a, theirs_b))


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x + 2.0,
        lambda x: 2.0 + x,
        lambda x: x - 2.0,
        lambda x: 2.0 - x,
        lambda x: x * 2.0,
        lambda x: 2.0 * x,
        lambda x: x / 2.0,
        lambda x: 2.0 / x,
        lambda x: x**2.5,
        lambda x: 2.0**x,
        lambda x: -x,
        lambda x: +x,
        lambda x: abs(x),
    ],
)
def test_array_with_constants(arrays, operation):
    ours, _, theirs, _ = arrays
    assert_array_same(operation(ours), operation(theirs))


def test_chained_expression(arrays):
    ours_a, ours_b, theirs_a, theirs_b = arrays
    ours = (ours_a * ours_b + mojo.sin(ours_a)) / mojo.sqrt(ours_b)
    theirs = (theirs_a * theirs_b + upstream.sin(theirs_a)) / upstream.sqrt(theirs_b)
    assert_array_same(ours, theirs)


def test_exact_array_correlation_cancellation(arrays):
    ours, _, theirs, _ = arrays
    result = ours - ours
    assert_array_same(result, theirs - theirs)
    assert np.array_equal(mojo.std_devs(result), np.zeros(ours.shape))


def test_reordered_shared_variables(arrays):
    ours, _, theirs, _ = arrays
    assert_array_same(ours + ours[::-1], theirs + theirs[::-1])
    assert_array_same(ours - ours[:, ::-1], theirs - theirs[:, ::-1])


def test_index_returns_correlated_scalar(arrays):
    ours, _, _, _ = arrays
    first = ours[0, 0]
    assert (first - ours[0, 0]).std_dev == 0.0
    assert first.nominal_value == 1.2
    assert first.std_dev == 0.02


def test_slicing_reshape_and_transpose(arrays):
    ours, _, theirs, _ = arrays
    assert_array_same(ours[:, 1:], theirs[:, 1:])
    assert_array_same(ours.reshape(3, 2), theirs.reshape(3, 2))
    assert_array_same(ours.T, theirs.T)


def test_broadcasting():
    nominal_a = np.array([[1.0], [2.0]])
    sigma_a = np.array([[0.1], [0.2]])
    nominal_b = np.array([3.0, 4.0, 5.0])
    sigma_b = np.array([0.3, 0.4, 0.5])
    ours_a = mojo.uarray(nominal_a, sigma_a)
    ours_b = mojo.uarray(nominal_b, sigma_b)
    theirs_a = upstream.uarray(nominal_a, sigma_a)
    theirs_b = upstream.uarray(nominal_b, sigma_b)
    assert_array_same(ours_a * ours_b + ours_a, theirs_a * theirs_b + theirs_a)


@pytest.mark.parametrize(
    ("name", "nominal"),
    [
        ("sin", np.linspace(-0.8, 0.8, 9)),
        ("cos", np.linspace(-0.8, 0.8, 9)),
        ("tan", np.linspace(-0.8, 0.8, 9)),
        ("arcsin", np.linspace(-0.8, 0.8, 9)),
        ("arccos", np.linspace(-0.8, 0.8, 9)),
        ("arctan", np.linspace(-0.8, 0.8, 9)),
        ("sinh", np.linspace(-0.8, 0.8, 9)),
        ("cosh", np.linspace(-0.8, 0.8, 9)),
        ("tanh", np.linspace(-0.8, 0.8, 9)),
        ("asinh", np.linspace(-0.8, 0.8, 9)),
        ("arccosh", np.linspace(1.1, 2.0, 9)),
        ("arctanh", np.linspace(-0.8, 0.8, 9)),
        ("exp", np.linspace(-0.8, 0.8, 9)),
        ("expm1", np.linspace(-0.8, 0.8, 9)),
        ("log", np.linspace(0.2, 2.0, 9)),
        ("log10", np.linspace(0.2, 2.0, 9)),
        ("log1p", np.linspace(-0.8, 0.8, 9)),
        ("sqrt", np.linspace(0.2, 2.0, 9)),
        ("erf", np.linspace(-0.8, 0.8, 9)),
        ("erfc", np.linspace(-0.8, 0.8, 9)),
        ("degrees", np.linspace(-0.8, 0.8, 9)),
        ("radians", np.linspace(-0.8, 0.8, 9)),
    ],
)
def test_unumpy_math(name, nominal):
    sigma = np.linspace(0.01, 0.09, nominal.size)
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    assert_array_same(getattr(mojo, name)(ours), getattr(upstream, name)(theirs))


@pytest.mark.parametrize(
    ("ufunc", "upstream_function"),
    [
        (np.sin, upstream.sin),
        (np.exp, upstream.exp),
        (np.sqrt, upstream.sqrt),
    ],
)
def test_numpy_ufunc_dispatch(ufunc, upstream_function):
    nominal = np.linspace(0.2, 1.0, 5)
    sigma = np.linspace(0.01, 0.05, 5)
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    assert_array_same(ufunc(ours), upstream_function(theirs))


@pytest.mark.parametrize("name", ["arctan2", "hypot", "copysign", "fmod"])
def test_unumpy_binary_math(name):
    nominal_a = np.linspace(0.7, 1.5, 7)
    nominal_b = np.linspace(2.0, 2.6, 7)
    sigma_a = np.linspace(0.01, 0.03, 7)
    sigma_b = np.linspace(0.04, 0.06, 7)
    ours_a = mojo.uarray(nominal_a, sigma_a)
    ours_b = mojo.uarray(nominal_b, sigma_b)
    theirs_a = upstream.uarray(nominal_a, sigma_a)
    theirs_b = upstream.uarray(nominal_b, sigma_b)
    assert_array_same(
        getattr(mojo, name)(ours_a, ours_b),
        getattr(upstream, name)(theirs_a, theirs_b),
        rtol=2e-8,
    )


@pytest.mark.parametrize("name", ["ceil", "floor", "trunc", "isinf", "isnan"])
def test_unumpy_locally_constant(name):
    nominal = np.array([-1.2, 0.0, 2.7])
    ours = mojo.uarray(nominal, [0.1, 0.2, 0.3])
    theirs = upstream.uarray(nominal, [0.1, 0.2, 0.3])
    assert np.array_equal(getattr(mojo, name)(ours), getattr(upstream, name)(theirs))


def test_unumpy_modf_and_ldexp():
    nominal = np.array([-1.2, 0.5, 2.7])
    sigma = np.array([0.1, 0.2, 0.3])
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    ours_parts = mojo.modf(ours)
    their_parts = upstream.modf(theirs)
    assert np.allclose(
        [part[0].nominal_value for part in ours_parts],
        [part[0].nominal_value for part in their_parts],
    )
    assert np.allclose(
        [part[0].std_dev for part in ours_parts],
        [part[0].std_dev for part in their_parts],
    )
    assert np.array_equal(
        [part[1] for part in ours_parts],
        [part[1] for part in their_parts],
    )
    assert_array_same(mojo.ldexp(ours, 3), upstream.ldexp(theirs, 3))


@pytest.mark.parametrize("name", ["gamma", "lgamma"])
def test_unumpy_object_fallback_math(name):
    nominal = np.array([1.2, 2.1, 3.4])
    sigma = np.array([0.01, 0.02, 0.03])
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    assert_array_same(getattr(mojo, name)(ours), getattr(upstream, name)(theirs), rtol=2e-8)


def test_scalar_uncertainty_broadcast_retains_correlation():
    scalar = ufloat(2.0, 0.1)
    values = mojo.uarray([1.0, 2.0, 3.0], [0.01, 0.02, 0.03])
    result = values * scalar - scalar
    expected_sigma = np.sqrt(
        np.array([0.01, 0.02, 0.03]) ** 2 * scalar.nominal_value**2
        + (np.array([1.0, 2.0, 3.0]) - 1) ** 2 * scalar.std_dev**2
    )
    assert np.allclose(mojo.std_devs(result), expected_sigma)


def test_object_array_conversion_matches_accessors(arrays):
    ours, _, _, _ = arrays
    objects = ours.to_object_array()
    assert objects.dtype == object
    assert np.allclose(mojo.nominal_values(objects), mojo.nominal_values(ours))
    assert np.allclose(mojo.std_devs(objects), mojo.std_devs(ours))


def test_plain_object_array_accessors():
    values = np.array([ufloat(1.0, 0.1), 2.0], dtype=object)
    assert np.array_equal(mojo.nominal_values(values), [1.0, 2.0])
    assert np.array_equal(mojo.std_devs(values), [0.1, 0.0])


def test_uarray_requires_two_arguments():
    with pytest.raises(TypeError, match="two arguments"):
        mojo.uarray([1.0, 2.0])


def test_empty_arrays_never_pass_null_or_dummy_buffers_to_ffi():
    values = mojo.uarray(np.empty((2, 0)), np.empty((2, 0)))
    result = mojo.sin(values + values)
    assert result.shape == (2, 0)
    assert mojo.std_devs(result).shape == (2, 0)


def test_strided_and_non_native_inputs_are_copied_safely():
    nominal = np.arange(17, dtype=">f8")[::-2]
    sigma = np.linspace(0.01, 0.09, nominal.size)[::-1]
    ours = mojo.uarray(nominal, sigma)
    theirs = upstream.uarray(nominal, sigma)
    assert_array_same(mojo.sin(ours * ours), upstream.sin(theirs * theirs))


@pytest.mark.parametrize("size", [131_071, 131_072, 131_073])
def test_parallel_partition_tails(size):
    nominal = np.linspace(0.2, 1.2, size)
    sigma = np.full(size, 0.01)
    result = mojo.sin(mojo.uarray(nominal, sigma))
    assert np.allclose(mojo.nominal_values(result), np.sin(nominal))
    assert np.allclose(mojo.std_devs(result), np.abs(np.cos(nominal) * sigma))


def test_unsupported_numeric_narrowing_is_explicit():
    with pytest.raises(TypeError, match="complex"):
        mojo.uarray(np.array([1 + 2j]), [0.1])
    with pytest.raises(TypeError, match="wider than float64"):
        mojo.uarray(np.array([1], dtype=np.longdouble), [0.1])
    with pytest.raises(OverflowError, match="exact float64"):
        mojo.uarray(np.array([2**53 + 1], dtype=np.int64), [1])
