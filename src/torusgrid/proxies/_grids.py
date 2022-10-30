from __future__ import annotations
from warnings import warn
import numpy as np

from ..core import FloatingPointPrecision

from ..grids import ComplexGrid


class ComplexGridNPZ:
    @staticmethod
    def read(path: str, **kwargs) -> ComplexGrid:
        with open(path, 'rb') as f:
            dat = np.load(f, **kwargs)

            psi = dat['psi']
            fft_axes = tuple(dat['fft_axes'].tolist())
            precision = FloatingPointPrecision.from_dtype(psi.dtype)

            shape = psi.shape
            grid = ComplexGrid(
                    shape, 
                    fft_axes=fft_axes,
                    precision=precision
            )

            if np.all(np.isreal(psi)):
                warn('Building ComplexGrid object with real data', np.ComplexWarning)

            grid.psi[...] = psi
        return grid

    @staticmethod
    def write(path: str, data: ComplexGrid, **kwargs) -> None:
        fft_axes = np.array(data.fft_axes, dtype=np.int_)
        with open(path, 'wb') as f:
            np.savez(f, 
                     psi=data.psi, 
                     fft_axes=fft_axes)



