from typing import TypeVar
import numpy as np
from ..fields import Field
from ..core.dtypes import FloatLike, FloatingPointPrecision


T = TypeVar('T', bound=Field)

def blend(
    field1: T, field2: T, *,
    axis: int,
    a: FloatLike = 0.25, 
    b: FloatLike = 0.75,
    interface_width: FloatLike
) -> T:
    """
    Blend two fields along an axis.

    Parameters:

        axis: axis along which to blend the fields
        a: 0~1
        b: a~1

    """
    assert 0 < a < b < 1

    precision = FloatingPointPrecision.most_precise(field1.precision, field2.precision)
    try:
        rtol = {
            'single': 1e-6,
            'double': 1e-12,
            'longdouble': 1e-18,
        }[precision.name.lower()]

        assert np.allclose(field1.size, field2.size, rtol=rtol, atol=0)
        assert field1.shape == field2.shape

    except AssertionError:
        raise ValueError(f'fields with different dimensions can not be combined using blend()')

    r1 = field1.r[axis]

    ra = field1.size[axis] * a
    rb = field1.size[axis] * b

    bump = (1+np.tanh((r1-ra)/interface_width))/2 * (1+np.tanh((-r1+rb)/interface_width))/2

    psi3 = field1.psi * bump + field2.psi * (1-bump)

    field3 = field1.copy()
    field3.psi[...] = psi3
    return field3




