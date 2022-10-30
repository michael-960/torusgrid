from __future__ import annotations

from typing import Optional, Tuple, final
from typing_extensions import Self

import numpy as np
import pyfftw



from ..core import get_complex_dtype

from ._base import Grid


class ComplexGrid(Grid[np.complexfloating]):
    """
    A ComplexGridND object is a complex array of shape (d1, d2, .., dN)
    equipped with fourier transform. No length scales are associated with the
    grid. 
    """
    @final
    def _init_grid_data(self, shape: Tuple[int, ...]):
        np.zeros
        self._psi = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))
        self._psi_k = pyfftw.zeros_aligned(shape, dtype=get_complex_dtype(self._precision))

    @property
    def isreal(self): return False

