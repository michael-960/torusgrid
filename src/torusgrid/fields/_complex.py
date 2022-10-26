from __future__ import annotations
import numpy as np


from ..core.fourier import generate_xk

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

        self._R.setflags(write=True)
        self._K.setflags(write=True)
        self._dR.setflags(write=True)
        self._dK.setflags(write=True)

        for i in range(self.rank):
            r, k, dr, dk = generate_xk(self.size[i], self.shape[i], precision=self.precision)
            R.append(r)
            K.append(k)
            DR.append(dr)
            DK.append(dk)

        self._R[...] = np.meshgrid(*R, indexing='ij')
        self._K[...] = np.meshgrid(*K, indexing='ij')
        self._dR[...] = np.array(DR)
        self._dK[...] = np.array(DK)

        self._R.setflags(write=False)
        self._K.setflags(write=False)
        self._dR.setflags(write=False)
        self._dK.setflags(write=False)


