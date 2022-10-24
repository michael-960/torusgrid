from __future__ import annotations
import numpy as np
from michael960lib.math import fourier

from ..grids import ComplexGrid
from ._base import Field


class ComplexField(Field[np.complexfloating], ComplexGrid):
    """
    A field is a grid with length scales. While the grid shape is fixed for a:
    particular instance, the system size (lengths in each dimension) can be
    changed with set_size().

    """
    def _update_coordinate_vars(self) -> None:
        R, K, DR, DK = [], [], [], []
        for i in range(self.rank):
            r, k, dr, dk = fourier.generate_xk(self.size[i], self.shape[i]) 
            R.append(r)
            K.append(k)
            DR.append(dr)
            DK.append(dk)
        

        self._R[...] = np.meshgrid(*R, indexing='ij')
        self._K[...] = np.meshgrid(*K, indexing='ij')
        self._dR[...] = np.array(DR)
        self._dK[...] = np.array(DK)


