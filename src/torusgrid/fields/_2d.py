from __future__ import annotations
from typing import Tuple, TypeVar, overload, Optional
from typing_extensions import Self
import numpy as np
import numpy.typing as npt

from ..core import FloatLike, SizeLike, PrecisionLike
from .. import core

from ..grids import Grid2D
from ._complex import ComplexField
from ._real import RealField
from ._base import Field

T = TypeVar('T', np.complexfloating, np.floating)

class Field2D(Field[T], Grid2D):

    @overload
    def __init__(
        self,
        lx: FloatLike, ly: FloatLike,
        nx: int, ny: int, /, *,
        precision: PrecisionLike = 'double',
        fft_axes: Optional[Tuple[int,...]]=None): ...

    @overload
    def __init__(
        self,
        size: SizeLike, shape: Tuple[int,int], /, *,
        precision: PrecisionLike = 'double',
        fft_axes: Optional[Tuple[int,...]]=None): ...

    def __init__(self, 
        arg1, arg2, arg3=None, arg4=None, /, *,
        precision: PrecisionLike = 'double',
        fft_axes: Optional[Tuple[int,...]]=None
    ):
        badargs = False

        if isinstance(arg2, tuple):
            size = arg1
            shape = arg2
            if (arg3 is not None) or (arg4 is not None): badargs = True
        else:
            size = (arg1, arg2)
            shape = (arg3, arg4)

        if not core.is_real_sequence(size, 2): badargs = True
        if not core.is_int_sequence(shape, 2): badargs = True

        if badargs:
            raise ValueError(f'Invalid size & shape arguments for 2D Field: {arg1}, {arg2}, {arg3}, {arg4}')
        

        super().__init__(
            size, shape, # type: ignore
            precision=precision, fft_axes=fft_axes
        )

    @property
    def lx(self) -> np.floating:
        return self.size[0]

    @property
    def ly(self) -> np.floating:
        return self.size[1]

    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]

    @property
    def x(self) -> npt.NDArray[np.floating]:
        return self.r[0]

    @property
    def y(self) -> npt.NDArray[np.floating]:
        return self.r[1]

    @property
    def dx(self) -> np.floating:
        return self.dr[0]

    @property
    def dy(self) -> np.floating:
        return self.dr[1]

    @property 
    def kx(self) -> npt.NDArray[np.floating]:
        return self.k[0]

    @property 
    def ky(self) -> npt.NDArray[np.floating]:
        return self.k[1]

    @property
    def dkx(self) -> np.floating:
        return self.dk[0]

    @property
    def dky(self) -> np.floating:
        return self.dk[1]

    @overload
    def set_size(self, lx: FloatLike, ly: FloatLike, /): ...
    @overload
    def set_size(self, size: SizeLike, /): ...

    def set_size(self, x1, x2=None, /):
        badargs = False 

        if core.is_real_sequence(x1, 2):
            if x2 is not None: badargs = True
            size = x1
        else:
            size = (x1,x2)

        if not core.is_real_sequence(size, 2): badargs = True
        
        if badargs:
            raise ValueError(f'Invalid size arguments for 1D field: {x1}, {x2}')

        Field[T].set_size(self, size) # type: ignore
            

class ComplexField2D(Field2D[np.complexfloating], ComplexField):
    """
    2D complex field
    """


class RealField2D(Field2D[np.floating], RealField):
    """
    2D real field
    """




