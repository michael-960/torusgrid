from __future__ import annotations
from typing import Tuple, TypeVar, overload
from typing_extensions import Self

import numpy as np

from torusgrid.core.dtypes import PrecisionLike


from ..core import PrecisionStr

from ._base import Grid
from ._complex import ComplexGrid
from ._real import RealGrid


T = TypeVar('T', np.complexfloating, np.floating)

class Grid1D(Grid[T]):
    @overload
    def __init__(self, n: int, /, *, precision: PrecisionLike = 'double'): ...
    @overload
    def __init__(self, shape: Tuple[int], /, *, precision: PrecisionLike = 'double'): ...

    def __init__(
        self, arg, /, *,
        precision: PrecisionLike = 'double',
        fft_axes=None
    ):
        badargs = False
        if fft_axes not in [None, (0,)]:
            raise ValueError('FFT axis for 1D grid should always be None or (0,)')

        if isinstance(arg, tuple):
            if len(arg) != 1: 
                badargs = True
            shape = arg
        else:
            if not isinstance(arg, (int, np.integer)):
                badargs = True
            shape = (arg,)

        if badargs:
            raise ValueError(f'Invalid shape argument for 1D grid: {arg}')

        super().__init__(shape, precision=precision, fft_axes=(0,))

    @property
    def n(self): 
        """=shape[0], the number of samples"""
        return self.shape[0]

    @property
    def N(self):
        """
        Deprecated; use self.n instead.
        """
        return self.n

    def copy(self) -> Self:
        g = self.__class__(self.n, precision=self._precision)
        g.psi[...] = self.psi
        return g


class ComplexGrid1D(ComplexGrid, Grid1D[np.complexfloating]):
    """
    1D complex grid; fft axis is always axis 0.
    """


class RealGrid1D(RealGrid, Grid1D[np.floating]):
    """
    1D real grid; fft axis is always axis 0.
    """


