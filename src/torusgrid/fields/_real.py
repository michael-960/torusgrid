from __future__ import annotations
from typing import Sequence, Tuple, Optional

import numpy as np
import numpy.typing as npt

from michael960lib.math import fourier
from torusgrid.typing.dtypes import PrecisionStr

from ..grids import RealGrid

from ..typing import SizeLike

from ._base import Field


class RealField(RealGrid, Field[np.floating]):
    def __init__(
            self, 
            size: Sequence[np.floating|float] | npt.NDArray[np.floating],
            shape: Tuple[int, ...], *,
            precision: PrecisionStr = 'double',
            fft_axes: Optional[Tuple[int,...]] = None
        ):
        super().__init__(shape, precision=precision, fft_axes=fft_axes)
        self._init_coordinate_vars()
        self.set_size(size)

    def _update_coordinate_vars(self) -> None:

        R, K, DR, DK = [], [], [], []
        for i in range(self.rank):
            if i == self.last_fft_axis:
                x, k, dx, dk = fourier.generate_xk(self.size[i], self.shape[i], real=True)
            else:
                x, k, dx, dk = fourier.generate_xk(self.size[i], self.shape[i]) 

            R.append(x)
            K.append(k)
            DR.append(dx)
            DK.append(dk)

        self._R[...] = np.meshgrid(*R, indexing='ij')
        self._K[...] = np.meshgrid(*K, indexing='ij')
        self._dR[...] = np.array(DR)
        self._dK[...] = np.array(DK)



