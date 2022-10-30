from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt

from ..core import PrecisionLike, FloatLike, SizeLike

from .. import core

from ..grids import Grid1D
from ._complex import ComplexField

from ._real import RealField
from ._base import Field

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar('T', np.complexfloating, np.floating)

class Field1D(Field[T], Grid1D):
    
    @overload
    def __init__(
        self,
        l: FloatLike, n: int, /, *,
        precision: PrecisionLike = 'double'
    ): ...

    @overload
    def __init__(
        self,
        size: SizeLike, shape: Tuple[int], /, *,
        precision: PrecisionLike = 'double'
    ): ...
    
    def __init__(
        self, 
        arg1, arg2, /, *,
        precision: PrecisionLike = 'double',
        fft_axes=None # for compatibility with base class
    ):
        badargs = False

        if fft_axes not in [None, (0,)]:
            raise ValueError('FFT axis for 1D field should always be None or (0,)')

        if isinstance(arg2, tuple):
            size = arg1
            shape = arg2
        else:
            size = (arg1,)
            shape = (arg2,)

        if not core.is_real_sequence(size, 1): badargs = True
        if not core.is_int_sequence(shape, 1): badargs = True


        if badargs:
            raise ValueError(f'Invalid size & shape arguments for 1D Field: {arg1}, {arg2}')

        super().__init__(
            size, shape,
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
    def set_size(self, size: SizeLike, /): ...

    def set_size(self, x):
        

        if core.is_real_sequence(x, 1):
            size = x
        else:
            size = (x,)

        if not core.is_real_sequence(size, 1):
            raise ValueError(f'Invalide size argument for 1D field: {x}')

        Field[T].set_size(self, size)




class ComplexField1D(ComplexField, Field1D[np.complexfloating]):
    """
    1D complex field
    """

class RealField1D(RealField, Field1D[np.floating]):
    """
    1D real field
    """


