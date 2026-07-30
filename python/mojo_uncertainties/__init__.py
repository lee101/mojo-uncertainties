"""First-order error propagation with Mojo-accelerated array arithmetic."""

from .core import (
    AffineScalarFunc,
    NegativeStdDev,
    UFloat,
    Variable,
    correlated_values,
    correlated_values_norm,
    correlation_matrix,
    covariance_matrix,
    modified_operators,
    modified_ops_with_reflection,
    nan_if_exception,
    nominal_value,
    std_dev,
    ufloat,
    ufloat_fromstr,
    wrap,
)

__version__ = "0.1.0"

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
