from __future__ import annotations
from typing import TypeVar
from typing_extensions import Self

import numpy as np


from ..core import PrecisionStr

from ._base import Grid
from ._complex import ComplexGrid
from ._real import RealGrid


T = TypeVar('T', np.complexfloating, np.floating)

class Grid1D(Grid[T]):
    def __init__(self,
            n: int, *,
            precision: PrecisionStr = 'double'):
        super().__init__((n,), precision=precision, fft_axes=(0,))

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

# class ComplexGrid1D(Grid1D[np.complexfloating], ComplexGrid):
class ComplexGrid1D(ComplexGrid, Grid1D[np.complexfloating]):
    """
    1D complex grid; fft axis is always axis 0.
    """


# class RealGrid1D(Grid1D[np.floating], RealGrid):
class RealGrid1D(RealGrid, Grid1D[np.floating]):
    """
    1D real grid; fft axis is always axis 0.
    """



