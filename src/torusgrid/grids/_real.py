from __future__ import annotations
from typing import Tuple, final

import numpy as np
from pyfftw import pyfftw
import warnings


from ..core import get_complex_dtype, get_real_dtype
from ._base import Grid


class RealGrid(Grid[np.floating]):
    """
    A RealGridND object is a real array of shape (d1, d2, .., dN) equipped
    with fourier transform. No length scales are associated with the grid.

    The generic type variables are the dtypes of self.psi & self.psi_k.
    """
    @final
    def _init_grid_data(self, shape: Tuple[int, ...]):

        last_axis = self._fft_axes[-1]

        if shape[last_axis] % 2 != 0:
            warnings.warn('odd resolution on the last axis will be automatically made even for RFFT')
            shape_ = list(shape)
            shape_[last_axis] += 1
            shape = tuple(shape_)

        shape_k = list(shape)
        shape_k[last_axis] = shape_k[last_axis]//2 + 1
        shape_k = tuple(shape_k)

        self._psi = pyfftw.zeros_aligned(shape, dtype=get_real_dtype(self._precision))
        self._psi_k = pyfftw.zeros_aligned(shape_k, dtype=get_complex_dtype(self._precision))

    @property
    @final
    def isreal(self): return True

