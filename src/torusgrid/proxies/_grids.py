from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from ..grids import ComplexGridND, RealGridND

if TYPE_CHECKING:
    from zlug.file.proxies import ObjectProxy


class ComplexGridH5:
    @staticmethod
    def read(path: str, **kwargs) -> ComplexGridND:
        raise NotImplementedError

    @staticmethod
    def write(path: str, data: ComplexGridND, **kwargs) -> None:
        raise NotImplementedError


class ComplexGridNPZ:
    @staticmethod
    def read(path: str, **kwargs) -> ComplexGridND:
        with open(path, 'rb') as f:
            dat = np.load(f) 
            psi = dat['psi']
            shape = psi.shape
            grid = ComplexGridND(shape)
            grid.set_psi(psi)
        return grid

    @staticmethod
    def write(path: str, data: ComplexGridND, **kwargs) -> None:
        with open(path, 'wb') as f:
            np.savez(f, psi=data.psi)

