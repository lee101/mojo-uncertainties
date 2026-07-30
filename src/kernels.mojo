"""Elementwise affine error-propagation kernels exposed through a C ABI."""

from std.algorithm import parallelize
from std.ffi import external_call
from std.math import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    expm1,
    log,
    log10,
    log1p,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)
from std.sys.info import num_physical_cores

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 131_072
comptime PI_OVER_180 = 0.017453292519943295
comptime INV_PI_OVER_180 = 57.29577951308232
comptime TWO_OVER_SQRT_PI = 1.1283791670955126


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def worker_count(n: Int) -> Int:
    if n < PARALLEL_THRESHOLD:
        return 1
    return max(num_physical_cores(), 1)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def libm_log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


def libm_log10(x: Float64) -> Float64:
    return external_call["log10", Float64](x)


def libm_log1p(x: Float64) -> Float64:
    return external_call["log1p", Float64](x)


def libm_pow(x: Float64, y: Float64) -> Float64:
    return external_call["pow", Float64](x, y)


def libm_erf(x: Float64) -> Float64:
    return external_call["erf", Float64](x)


def libm_erfc(x: Float64) -> Float64:
    return external_call["erfc", Float64](x)


def libm_atan2(y: Float64, x: Float64) -> Float64:
    return external_call["atan2", Float64](y, x)


def libm_hypot(x: Float64, y: Float64) -> Float64:
    return external_call["hypot", Float64](x, y)


def libm_copysign(x: Float64, y: Float64) -> Float64:
    return external_call["copysign", Float64](x, y)


def libm_fmod(x: Float64, y: Float64) -> Float64:
    return external_call["fmod", Float64](x, y)


def libm_trunc(x: Float64) -> Float64:
    return external_call["trunc", Float64](x)


def binary_point(op: Int, x: Float64, y: Float64) -> Tuple[Float64, Float64, Float64]:
    if op == 0:
        return x + y, 1.0, 1.0
    if op == 1:
        return x - y, 1.0, -1.0
    if op == 2:
        return x * y, y, x
    if op == 3:
        return x / y, 1.0 / y, -x / (y * y)
    if op == 4:
        var value = libm_pow(x, y)
        var dx = nan_value()
        var dy = nan_value()
        if x > 0.0 or (y % 1.0 == 0.0 and (x < 0.0 or y >= 1.0)):
            dx = y * libm_pow(x, y - 1.0)
        elif x == 0.0 and y == 0.0:
            dx = 0.0
        if x > 0.0:
            dy = libm_log(x) * value
        elif x == 0.0 and y > 0.0:
            dy = 0.0
        return value, dx, dy
    if op == 5:
        var denominator = x * x + y * y
        return libm_atan2(x, y), y / denominator, -x / denominator
    if op == 6:
        var value = libm_hypot(x, y)
        return value, x / value, y / value
    if op == 7:
        var derivative = libm_copysign(1.0, y) if x >= 0.0 else -libm_copysign(1.0, y)
        return libm_copysign(x, y), derivative, 0.0
    return libm_fmod(x, y), 1.0, -libm_trunc(x / y)


def unary_point(op: Int, x: Float64) -> Tuple[Float64, Float64]:
    if op == 0:
        return -x, -1.0
    if op == 1:
        return x, 1.0
    if op == 2:
        return abs(x), 1.0 if x >= 0.0 else -1.0
    if op == 3:
        var value = exp(x)
        return value, value
    if op == 4:
        return expm1(x), exp(x)
    if op == 5:
        return libm_log(x), 1.0 / x
    if op == 6:
        return libm_log10(x), 1.0 / (x * libm_log(10.0))
    if op == 7:
        return libm_log1p(x), 1.0 / (1.0 + x)
    if op == 8:
        var value = sqrt(x)
        return value, 0.5 / value
    if op == 9:
        return sin(x), cos(x)
    if op == 10:
        return cos(x), -sin(x)
    if op == 11:
        var value = tan(x)
        return value, 1.0 + value * value
    if op == 12:
        return asin(x), 1.0 / sqrt(1.0 - x * x)
    if op == 13:
        return acos(x), -1.0 / sqrt(1.0 - x * x)
    if op == 14:
        return atan(x), 1.0 / (1.0 + x * x)
    if op == 15:
        return sinh(x), cosh(x)
    if op == 16:
        return cosh(x), sinh(x)
    if op == 17:
        var value = tanh(x)
        return value, 1.0 - value * value
    if op == 18:
        return asinh(x), 1.0 / sqrt(1.0 + x * x)
    if op == 19:
        return acosh(x), 1.0 / sqrt(x * x - 1.0)
    if op == 20:
        return atanh(x), 1.0 / (1.0 - x * x)
    if op == 21:
        return libm_erf(x), exp(-(x * x)) * TWO_OVER_SQRT_PI
    if op == 22:
        return libm_erfc(x), -exp(-(x * x)) * TWO_OVER_SQRT_PI
    if op == 23:
        return x * INV_PI_OVER_180, INV_PI_OVER_180
    return x * PI_OVER_180, PI_OVER_180


@export("munc_binary")
def munc_binary(
    op: Int,
    a_addr: Int,
    b_addr: Int,
    value_addr: Int,
    da_addr: Int,
    db_addr: Int,
    n: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var values = fp(value_addr)
    var da = fp(da_addr)
    var db = fp(db_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            var point = binary_point(op, a[i], b[i])
            values[i] = point[0]
            da[i] = point[1]
            db[i] = point[2]

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_unary")
def munc_unary(
    op: Int,
    x_addr: Int,
    value_addr: Int,
    derivative_addr: Int,
    n: Int,
) abi("C"):
    var x = fp(x_addr)
    var values = fp(value_addr)
    var derivative = fp(derivative_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            var point = unary_point(op, x[i])
            values[i] = point[0]
            derivative[i] = point[1]

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_chain_binary")
def munc_chain_binary(
    da_addr: Int,
    db_addr: Int,
    ga_addr: Int,
    gb_addr: Int,
    dst_addr: Int,
    n: Int,
) abi("C"):
    var da = fp(da_addr)
    var db = fp(db_addr)
    var ga = fp(ga_addr)
    var gb = fp(gb_addr)
    var destination = fp(dst_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            destination[i] = da[i] * ga[i] + db[i] * gb[i]

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_chain_unary")
def munc_chain_unary(
    local_addr: Int,
    gradient_addr: Int,
    dst_addr: Int,
    n: Int,
) abi("C"):
    var local = fp(local_addr)
    var gradient = fp(gradient_addr)
    var destination = fp(dst_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            destination[i] = local[i] * gradient[i]

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_variance_diag")
def munc_variance_diag(
    gradient_addr: Int,
    sigma_addr: Int,
    variance_addr: Int,
    n: Int,
) abi("C"):
    var gradient = fp(gradient_addr)
    var sigma = fp(sigma_addr)
    var variance = fp(variance_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            var contribution = gradient[i] * sigma[i]
            variance[i] += contribution * contribution

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_variance_cross")
def munc_variance_cross(
    gradient_a_addr: Int,
    sigma_a_addr: Int,
    ids_a_addr: Int,
    gradient_b_addr: Int,
    sigma_b_addr: Int,
    ids_b_addr: Int,
    variance_addr: Int,
    n: Int,
) abi("C"):
    var gradient_a = fp(gradient_a_addr)
    var sigma_a = fp(sigma_a_addr)
    var ids_a = ip(ids_a_addr)
    var gradient_b = fp(gradient_b_addr)
    var sigma_b = fp(sigma_b_addr)
    var ids_b = ip(ids_b_addr)
    var variance = fp(variance_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            if ids_a[i] == ids_b[i]:
                variance[i] += (
                    2.0
                    * gradient_a[i]
                    * sigma_a[i]
                    * gradient_b[i]
                    * sigma_b[i]
                )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("munc_std_finish")
def munc_std_finish(variance_addr: Int, diagonal_addr: Int, n: Int) abi("C"):
    var variance = fp(variance_addr)
    var diagonal = fp(diagonal_addr)
    var workers = worker_count(n)

    @parameter
    def process(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        for i in range(start, end):
            if abs(variance[i]) <= 1.0e-14 * diagonal[i]:
                variance[i] = 0.0
            variance[i] = sqrt(max(variance[i], 0.0))

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)
