from __future__ import annotations
from typing import Optional, Tuple, TypeVar
from typing_extensions import Self

import numpy as np

from ..core import PrecisionStr
from ._base import Grid
from ._complex import ComplexGrid
from ._real import RealGrid


T = TypeVar('T', np.complexfloating, np.floating)

class Grid2D(Grid[T]):
    def __init__(self, 
        nx: int, ny: int, *,
        precision: PrecisionStr = 'double', 
        fft_axes: Optional[Tuple[int,...]]=None):

        super().__init__(
                (nx, ny),
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


# class ComplexGrid2D(Grid2D[np.complexfloating], ComplexGrid):
class ComplexGrid2D(ComplexGrid, Grid2D[np.complexfloating]):
    """
    2D complex grid
    """


# class RealGrid2D(Grid2D[np.floating], RealGrid):
class RealGrid2D(RealGrid, Grid2D[np.floating]):
    """
    2D real grid
    """

