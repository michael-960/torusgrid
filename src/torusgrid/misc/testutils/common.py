from __future__ import annotations
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, TypeVar, overload
import yaml
from pathlib import Path
import pytest
import numpy as np

from torusgrid.core.dtypes import FloatLike


from ...core import PrecisionLike, PrecisionStr, FloatingPointPrecision, get_real_dtype

from ... import grids
from ... import fields



def prod(arr: Sequence[float], precision: PrecisionLike = 'double'):
    """product of all the elements in an array"""
    r = get_real_dtype(precision)('1.0')
    for a in arr: r = r * a
    return r

@overload
def get_class(real: bool, field: Literal[True], explicit_rank: int) -> Type[fields.Field]: ...

@overload
def get_class(real: bool, field: Literal[False], explicit_rank: int) -> Type[grids.Grid]: ...

@overload
def get_class(real: bool, field: bool, explicit_rank: int) -> Type[grids.Grid]|Type[fields.Field]: ...

def get_class(real: bool, field: bool, explicit_rank: int) -> Type[grids.Grid]|Type[fields.Field]:
    """
    Get a Grid/Field class
    """
    return {
        True: {
            True: {
                0: fields.RealField,
                1: fields.RealField1D,
                2: fields.RealField2D,
            },
            False: {
                0: grids.RealGrid,
                1: grids.RealGrid1D,
                2: grids.RealGrid2D,
            }
        },
        False: {
            True: {
                0: fields.ComplexField,
                1: fields.ComplexField1D,
                2: fields.ComplexField2D,
            },
            False: {
                0: grids.ComplexGrid,
                1: grids.ComplexGrid1D,
                2: grids.ComplexGrid2D,
            }
        }
    }[real][field][explicit_rank]



def get_config(filename: str = 'config.yaml', *, max_depth=3) -> Dict[str,Any]:

    dir = Path('.')

    for _ in range(max_depth):
        path = dir / filename

        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            pass
        
        dir = dir / '..'

    raise FileNotFoundError(
            f'Filename {filename} is not found within search depth {max_depth}'
            )



def even_shape(shape: Tuple[int,...], fft_axes: Tuple[int,...]):
    shape_ = list(shape)
    if shape_[fft_axes[-1]] % 2 == 1:
        shape_[fft_axes[-1]] += 1
    shape = tuple(shape_)
    return shape


def will_overflow(size: Tuple[FloatLike,...], shape: Tuple[int,...], precision: PrecisionLike):
    l = min(size)
    n = max(shape)
    dk = np.pi*2 / l
    k = dk * n / 2

    p = FloatingPointPrecision.cast(precision)
        
    if p is FloatingPointPrecision.SINGLE:
        return k**6 > 1e38 or k**6 < 1e-38

    if p is FloatingPointPrecision.DOUBLE:
        return k**6 > 1e300 or k**6 < 1e-300

    if p is FloatingPointPrecision.LONGDOUBLE:
        return k**6 > np.longdouble('1e4000') or k**6 < np.longdouble('1e-4000')

    raise ValueError(f'Invalid precision: {precision}')



def get_grid(
        real: bool, 
        shape: Tuple[int,...], 
        size: Tuple[float,...]|None,
        fft_axes: Tuple[int,...], 
        prec: PrecisionStr,
        explicit_rank: int = 0
        ) -> grids.Grid:

    isfield = (size is not None)

    cls = get_class(real, isfield, explicit_rank)

    if size is not None:
        if will_overflow(size, shape, prec):
            pytest.skip()

    if real: shape = even_shape(shape, fft_axes)

    if isfield:
        assert issubclass(cls, fields.Field)
        g = cls(size, shape, fft_axes=fft_axes, precision=prec)
    else:
        assert not issubclass(cls, fields.Field)
        g = cls(shape, fft_axes=fft_axes, precision=prec)

    return g

