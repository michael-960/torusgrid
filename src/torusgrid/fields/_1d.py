from __future__ import annotations
from abc import abstractmethod

from typing import Tuple, TypeVar, overload
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt

from ..typing.dtypes import PrecisionStr
from ..typing import FloatLike


from ..grids import ComplexGrid, Grid1D, Grid2D
from ._complex import ComplexField

from ._real import RealField
from ._base import Field

T = TypeVar('T', np.complexfloating, np.floating)

class Field1D(Field[T], Grid1D):
    def __init__(
            self, 
            l: FloatLike, n: int, *,
            precision: PrecisionStr = 'double'):

        super().__init__(
            (l,), (n,),
            precision=precision, fft_axes=(0,)
        )

    @property
    def l(self) -> np.floating:
        return self.size[0]

    @property
    def n(self):
        return self.shape[0]

    @property
    def x(self) -> npt.NDArray[np.floating]:
        return self.r[0]

    @property
    def dx(self) -> np.floating:
        return self.dr[0]

    @property 
    def kx(self) -> npt.NDArray[np.floating]:
        return self.k[0]

    @property
    def dkx(self) -> np.floating:
        return self.dk[0]

    @overload
    def set_size(self, l: FloatLike, /): ...
    
    @overload
    def set_size(self, size: Tuple[FloatLike], /): ...

    def set_size(self, x: Tuple[FloatLike]|FloatLike, /):
        if isinstance(x, tuple): 
            super().set_size(x)

        else:
            super().set_size((x,))

    def copy(self) -> Self:
        f = self.__class__(
                self.l, self.n,
                precision=self._precision
            )
        return f


class ComplexField1D(Field1D[np.complexfloating], ComplexField):
    '''
    1D complex field
    '''


class RealField1D(Field1D[np.floating], RealField):
    '''
    1D real field
    '''


