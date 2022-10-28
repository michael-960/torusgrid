"""
Utility functions used for testing
"""

from .asserts import (adaptive_allclose, same_dimensions, same_meta,
                      same_meta_except_dimensions, distance)

from .common import prod, get_class, get_config, get_grid, will_overflow, even_shape


from .dtype import floating_tol, real_dtype, complex_dtype, nbytes


from .gen import gen_array, gen_fft_axes, gen_random_sequence, gen_shapes, random_permutation


from .pytest import param_repr, params_repr, float_tup_repr, int_tup_repr, parametrize
