from __future__ import annotations
from typing import Sequence, Tuple, Optional
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from michael960lib.math import fourier
from torusgrid.typing.dtypes import PrecisionStr

from ..grids import RealGrid

from ..typing import SizeLike

from ._base import Field


class RealField(Field[np.floating], RealGrid):
    def _update_coordinate_vars(self) -> None:
        R, K, DR, DK = [], [], [], []
        for i in range(self.rank):
            if i == self.last_fft_axis:
                r, k, dr, dk = fourier.generate_xk(self.size[i], self.shape[i], real=True)
            else:
                r, k, dr, dk = fourier.generate_xk(self.size[i], self.shape[i]) 

            R.append(r)
            K.append(k)
            DR.append(dr)
            DK.append(dk)

        self._R[...] = np.meshgrid(*R, indexing='ij')
        self._K[...] = np.meshgrid(*K, indexing='ij')
        self._dR[...] = np.array(DR)
        self._dK[...] = np.array(DK)


    
