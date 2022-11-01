from __future__ import annotations
from warnings import warn

from typing import Literal, Optional, Sequence

from ..core import FloatLike, ComplexLike, PrecisionLike, FloatingPointPrecision
from ..grids import Grid

from typing import Tuple, TypeVar
import numpy as np

import skimage.transform as skt

T = TypeVar('T', bound=Grid)


def extend(grid: T, factors: Tuple[int,...]) -> T:
    """
    Extend the grid periodically over all axes

    Parameters:
        x: Any grid
        factors: (factor0, factor1, ...) are the integer factors by which each
                   axis is extended
    """
    if grid.rank != len(factors):
        raise ValueError(f'Received factors {factors} but input grid has rank {grid.rank}')

    cls = grid.__class__

    new_psi = grid.psi.copy()
    newmeta = grid.metadata().copy()
    for axis, m in enumerate(factors):
        new_psi = np.concatenate([new_psi for _ in range(m)], axis=axis, dtype=grid.psi.dtype)
        newmeta = cls.concat(*[newmeta for _ in range(m)], axis=axis)

    y = cls.from_array(new_psi, metadata=newmeta)
    return y


def transpose(grid: T, axes: Sequence[int]) -> T:
    """
    Transpose the axes of a grid

    Parameters:
        grid: Any grid
        axes: A list of integers specifying a permutation of x's axes

    Example:
        If x is a 2D Grid, transpose(x, 1, 0) will flip the X and Y axes

    """
    cls = grid.__class__
    if sorted(tuple(axes)) != sorted(range(grid.rank)):
        raise ValueError(f'Received axes {axes} but input grid has rank {grid.rank}')

    newpsi = np.transpose(grid.psi.copy(), axes)
    newmeta = cls.transpose(grid.metadata(), tuple(axes))
    y = cls.from_array(newpsi, metadata=newmeta)
    return y


def concat(*grids: T, axis: int) -> T:
    """
    Concatenate two grids along an axis
    """
    clss = [type(grid) for grid in grids]
    if clss[1:] != clss[:-1]:
        raise ValueError('input grids must be of the same type')
    newpsi = np.concatenate([grid.psi for grid in grids], axis=axis)
    newmeta = clss[0].concat(*[grid.metadata() for grid in grids], axis=axis)
    z = clss[0].from_array(newpsi, metadata=newmeta)
    return z


def const_like(grid: T, fill: Optional[FloatLike|ComplexLike]=None) -> T:
    """
    Return a new grid with constant value
    
    Parameters:
        x: Any grid
        fill: The constant fill value. If not specified, the mean of the grid's
              data will be used instead
    """

    if fill is None:
        fill_value = grid.psi.mean()
    else:
        fill_value = fill

    y = grid.copy()
    y.psi[...] = fill_value

    return y


def flip(grid: T, axes: Sequence[int]) -> T:

    if len(axes) != len(set(axes)):
        raise ValueError(f'Duplicate axes: {axes}')

    if max(axes) >= grid.rank:
        raise ValueError(f'Axes out of range: {axes}')

    y = grid.copy()

    y.psi[...] = np.flip(y.psi, axis=axes)

    return y


def resample(grid: T, shape: Tuple[int,...], *, 
             order: int, 
             mode: Literal['constant','edge','symmetric','reflect','wrap']='reflect',
             cval: int = 0,
             clip: bool = True,
             preserve_range: bool = False,
             anti_aliasing: bool = False

             ) -> T:
    """
    Change the shape and interpolate data
    """
    
    if len(shape) != grid.rank:
        raise ValueError('output shape must be of the same rank as original shape')

    cls = type(grid)
    metadata = grid.metadata()
    metadata['shape'] = shape

    if not grid.isreal:
        psi_re = np.real(grid.psi)
        psi_im = np.imag(grid.psi)

        newpsi_re = skt.resize(
                        psi_re, 
                        shape, order=order, mode=mode,
                        cval=cval, clip=clip, preserve_range=preserve_range,
                        anti_aliasing=anti_aliasing)

        newpsi_im = skt.resize(
                        psi_im,
                        shape, order=order, mode=mode,
                        cval=cval, clip=clip, preserve_range=preserve_range,
                        anti_aliasing=anti_aliasing)

        newpsi = newpsi_re + 1j*newpsi_im

    else:
        newpsi = skt.resize(grid.psi, shape, 
                            order=order, mode=mode,
                            cval=cval, clip=clip, preserve_range=preserve_range,
                            anti_aliasing=anti_aliasing)

    return cls.from_array(newpsi, metadata=metadata)


def change_precision(grid: T, precision: PrecisionLike) -> T:
    """
    Change the precision
    """
    cls = type(grid)
    metadata = grid.metadata().copy()
    metadata['precision'] = FloatingPointPrecision.cast(precision).name

    return cls.from_array(grid.psi, metadata=metadata)


def crop(grid: T, axis: int, a: int, b: int) -> T:
    """
    Crop input grid along the given axis 

    Parameters:
        a: start index (inclusive)
        b: end index (exclusive)
    """
    cls = type(grid)
    
    slices = tuple(slice(None) if a != axis else slice(a,b) for a in range(grid.rank))
    newpsi = grid.psi.copy()[slices]

    newmeta = cls.crop(grid.metadata.copy(), axis, a, b)

    return cls.from_array(newpsi, metadata=newmeta)







