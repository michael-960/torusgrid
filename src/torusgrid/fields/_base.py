from __future__ import annotations
from typing import Tuple
import shutil

import numpy as np

from michael960lib.math import fourier
from michael960lib.common import overrides
from ..grids import ComplexGrid2D, ComplexGridND, RealGridND



class ComplexFieldND(ComplexGridND):
    def __init__(self, size: Tuple[float, ...], shape: Tuple[int, ...]):
        super().__init__(shape)
        self.set_size(size)

    def set_dimensions(self, size: Tuple[float, ...], shape: Tuple[int, ...]):
        self.set_resolution(shape)
        self.set_size(size)

    def set_size(self, size: Tuple[float, ...]):
        if len(size) != self.rank:
            raise ValueError(f'size {size} is incompatible with current shape {self.shape}')
        self.size = size
        self.Volume = np.prod(size)
        
        R, K, DR, DK = [], [], [], []

        for i in range(self.rank):
            x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i]) 
            R.append(x)
            K.append(k)
            DR.append(dx)
            DK.append(dk)

        self.R = np.meshgrid(*R, indexing='ij')
        self.K = np.meshgrid(*K, indexing='ij')
        self.dR = np.array(DR)
        self.dK = np.array(DK)

        self.dV = np.prod(self.dR)

    @overrides(ComplexGridND)
    def export_state(self) -> dict:
        state = {'psi': self.psi.copy(), 'size': self.size}
        return state

    @overrides(ComplexGridND)
    def copy(self):
        field1 = self.__class__(self.size, self.shape)
        field1.set_psi(self.psi)
        return field1

    @overrides(ComplexGrid2D)
    def save(self, fname: str, verbose=False):
        tmp_name = f'{fname}.tmp.file'
        if verbose:
            self.yell(f'dumping field data to {fname}.field')
        np.savez(tmp_name, **self.export_state()) 
        shutil.move(f'{tmp_name}.npz', f'{fname}.field')


class RealFieldND(ComplexFieldND, RealGridND):
    def __init__(self, size: Tuple[float, ...], shape: Tuple[int, ...]):
        super().__init__(size, shape)
        self._isreal = True

    @overrides(ComplexFieldND)
    def set_size(self, size: Tuple[float, ...]):
        if len(size) != self.rank:
            raise ValueError(f'size {size} is incompatible with current shape {self.shape}')
        self.size = size
        self.Volume = np.prod(size)
        
        R, K, DR, DK = [], [], [], []

        for i in range(self.rank):
            if i == self.rank - 1:
                x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i], real=True)
            else:
                x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i]) 

            R.append(x)
            K.append(k)
            DR.append(dx)
            DK.append(dk)

        self.R = np.meshgrid(*R, indexing='ij')
        self.K = np.meshgrid(*K, indexing='ij')
        self.dR = np.array(DR)
        self.dK = np.array(DK)

        self.dV = np.prod(self.dR)



