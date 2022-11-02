
from .consts import pi, e

from .dtypes import (PrecisionLike, PrecisionStr, 
                     NPFloat, NPComplex, FloatLike, ComplexLike,
                     NPInt, IntLike,
                     SizeLike,
                     FloatingPointPrecision,
                     get_real_dtype, get_complex_dtype, get_dtype,
                     is_real_scalar, is_int_scalar,
                     is_real_sequence, is_int_sequence,
                     float_fmt, highlight_last_digits
                     )

from .fourier import generate_xk, FFTWEffort

from .data import scalarize

from .generics import generic

from .version import __version__
