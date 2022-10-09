from __future__ import annotations

from typing import Tuple, overload, Optional

import numpy as np
import numpy.typing as npt


from ..typing.dtypes import PrecisionStr
from ..typing import FloatLike


from ..grids import Grid2D
from ._complex import ComplexField
from ._base import Field

class Field2D(Grid2D, Field):
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
    def set_size(self, size: Tuple[FloatLike, FloatLike], /): ...

    def set_size(self, 
            x1: Tuple[FloatLike, FloatLike]|FloatLike, 
            x2: Optional[FloatLike]=None, /):

        if isinstance(x1, tuple): 
            assert x2 is None
            super().set_size(x1)
        else:
            assert x2 is not None
            super().set_size((x1, x2))


class ComplexField2D(Field2D, ComplexField):
    def __init__(self, 
            lx: FloatLike, ly: FloatLike,
            nx: int, ny: int, *,
            precision: PrecisionStr,
            fft_axes: Optional[Tuple[int,...]]=None
            ):
        super().__init__(
                (lx, ly), (nx, ny),
                precision=precision, fft_axes=fft_axes
        )


class RealField2D(Field2D, ComplexField):
    def __init__(self, 
            lx: FloatLike, ly: FloatLike,
            nx: int, ny: int, *,
            precision: PrecisionStr,
            fft_axes: Optional[Tuple[int,...]]=None
            ):
        super().__init__(
                (lx, ly), (nx, ny),
                precision=precision, fft_axes=fft_axes
        )


