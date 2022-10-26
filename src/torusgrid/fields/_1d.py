from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt

from ..core import PrecisionLike, FloatLike

from ..grids import Grid1D
from ._complex import ComplexField

from ._real import RealField
from ._base import Field

if TYPE_CHECKING:
    from typing_extensions import Self

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
            Field[T].set_size(self, x)

        else:
            Field[T].set_size(self, (x,))

    def copy(self) -> Self:
        f = self.__class__(
                self.l, self.n,
                precision=self._precision
            )

        f.psi[...] = self.psi
        return f


#class ComplexField1D(Field1D[np.complexfloating], ComplexField):
class ComplexField1D(ComplexField, Field1D[np.complexfloating]):
    """
    1D complex field
    """

# class RealField1D(Field1D[np.floating], RealField):
class RealField1D(RealField, Field1D[np.floating]):
    """
    1D real field
    """

