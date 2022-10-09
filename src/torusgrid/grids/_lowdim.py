from __future__ import annotations
from typing import Optional, Tuple, TypeVar

import numpy as np

from torusgrid.typing.dtypes import PrecisionStr
from ._base import Grid
from ._complex import ComplexGrid
from ._real import RealGrid


T = TypeVar('T', np.complexfloating, np.floating)


class Grid1D(Grid):
    @property
    def n(self): return self.shape[0]

    @property
    def N(self):
        '''
        Deprecated; use self.n instead.
        '''
        return self.n


class ComplexGrid1D(Grid1D, ComplexGrid):
    '''
    1D complex grid; fft axis is always axis 0.
    '''

    def __init__(self, 
            n: int, *,
            precision: PrecisionStr = 'double'):

        super().__init__(
            (n,),
            precision=precision,
            fft_axes=(0,))


class RealGrid1D(Grid1D, RealGrid):
    '''
    1D real grid; fft axis is always axis 0.
    '''

    def __init__(self, 
            n: int, *,
            precision: PrecisionStr = 'double'):

        super().__init__(
                (n,),
                precision=precision,
                fft_axes=(0,))


class Grid2D(Grid):
    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]


class ComplexGrid2D(Grid2D, ComplexGrid):
    '''
    2D complex grid
    '''

    def __init__(self, 
            nx: int, ny: int, *,
            precision: PrecisionStr = 'double', 
            fft_axes: Optional[Tuple[int,...]]=None
            ):
        super().__init__(
                (nx, ny),
                precision=precision,
                fft_axes=fft_axes)

class RealGrid2D(Grid2D, RealGrid):
    '''
    2D real grid
    '''
    def __init__(self,
            nx: int, ny: int, *,
            precision: PrecisionStr = 'double', 
            fft_axes: Optional[Tuple[int,...]]=None
            ):
        super().__init__(
                (nx, ny),
                precision=precision,
                fft_axes=fft_axes)


        

