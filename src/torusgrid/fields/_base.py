from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, Tuple, Type, TypeVar
import shutil
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from michael960lib.math import fourier
from michael960lib.common import overrides
from ..grids import ComplexGridND, RealGridND




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
    def copy(self) -> Self:
        field1 = self.__class__(self.size, self.shape)
        field1.set_psi(self.psi)
        return field1

    @overrides(ComplexGridND)
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



T = TypeVar('T', bound=ComplexFieldND)

class FreeEnergyFunctional(ABC, Generic[T]):
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def free_energy_density(self, field: T) -> npt.NDArray:
        raise NotImplementedError()

    def free_energy(self, field: T) -> float:
        return np.sum(self.free_energy_density(field)) * field.dV

    @abstractmethod
    def derivative(self, field: T):
        raise NotImplementedError()

    def mean_free_energy_density(self, field: T) -> float:
        return np.mean(self.free_energy_density(field))




