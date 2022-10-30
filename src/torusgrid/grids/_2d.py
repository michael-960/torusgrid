from __future__ import annotations
from typing import Optional, Tuple, TypeVar, overload
from typing_extensions import Self

import numpy as np

from torusgrid.core.dtypes import PrecisionLike


from ._base import Grid
from ._complex import ComplexGrid
from ._real import RealGrid


T = TypeVar('T', np.complexfloating, np.floating)

class Grid2D(Grid[T]):
    @overload
    def __init__(self, nx: int, ny: int, /, *, 
                 precision: PrecisionLike = 'double', 
                 fft_axes: Optional[Tuple[int,...]]=None): ...

    @overload
    def __init__(self, shape: Tuple[int,int], /, *,
                 precision: PrecisionLike = 'double', 
                 fft_axes: Optional[Tuple[int,...]]=None): ...

    def __init__(
        self, arg1, arg2=None, /, *,
        precision: PrecisionLike = 'double', 
        fft_axes: Optional[Tuple[int,...]]=None):

        badargs = False

        if isinstance(arg1, tuple):
            if arg2 is not None:
                badargs = True

            if len(arg1) != 2:
                badargs = True
            shape = arg1
        
        else:
            if not isinstance(arg1, (np.integer, int)) or not isinstance(arg2, (np.integer, int)):
                badargs = True
            shape = (arg1, arg2)

        if badargs:
            raise ValueError(f'Invalid shape arguments for 2D grid: {arg1}, {arg2}')
       
        super().__init__(
                shape, # type: ignore
                precision=precision,
                fft_axes=fft_axes)

    @property
    def nx(self):
        """
        = shape[0], the number of samples in the x direction
        """
        return self.shape[0]

    @property
    def ny(self):
        """
        = shape[1], the number of samples in the y direction
        """
        return self.shape[1]

    def copy(self) -> Self:
        g = self.__class__(
                self.nx, self.ny,
                precision=self.precision,
                fft_axes=self.fft_axes)

        g.psi[...] = self.psi
        return g



class ComplexGrid2D(ComplexGrid, Grid2D[np.complexfloating]):
    """
    2D complex grid
    """

class RealGrid2D(RealGrid, Grid2D[np.floating]):
    """
    2D real grid
    """


