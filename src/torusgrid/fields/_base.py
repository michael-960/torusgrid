from __future__ import annotations
from abc import abstractmethod
from typing import TypeVar, final

import numpy as np
import numpy.typing as npt

from torusgrid.typing.general import NPFloat

from ..typing import get_real_dtype, SizeLike

from ..grids import Grid



T = TypeVar('T', np.complexfloating, np.floating)

class Field(Grid[T]):
    '''
    Base class for fields.
    '''

    _size: npt.NDArray[np.floating]

    _R: npt.NDArray[np.floating]
    _K: npt.NDArray[np.floating]

    _dR: npt.NDArray[np.floating]
    _dK: npt.NDArray[np.floating]

    _volume: NPFloat
    _dV: NPFloat

    @final
    def _init_coordinate_vars(self):
        dtype = get_real_dtype(self._precision)

        self._size = np.zeros((self.rank,), dtype=dtype)

        self._R = np.zeros((self.rank, *self.shape), dtype=dtype)


        k_shape = list(self.shape)
        if self._isreal:
            k_shape[self.last_fft_axis] = k_shape[self.last_fft_axis] // 2 + 1

        self._K = np.zeros((self.rank, *k_shape), dtype=dtype)
        self._dR = np.zeros((self.rank,), dtype=dtype)
        self._dK = np.zeros((self.rank,), dtype=dtype)

    @abstractmethod
    def _update_coordinate_vars(self) -> None:
        '''
        Given size, update _R, _K, _dR, _dK
        '''

    def set_size(self, size: SizeLike):
        '''
        Set system size (dimension lengths)
        '''
        self.validate_size(size)

        self._size[...] = size

        self._update_coordinate_vars()

        self._volume = np.prod(self._size)
        self._dV = np.prod(self._dR)

    def validate_size(self, size: SizeLike):
        if isinstance(size, np.ndarray):
            if size.ndim != 1:
                raise ValueError(f'ndarray with shape {size.shape} is not a valid field size')

            if not np.isreal(size):
                raise ValueError(f'ndarray with dtype {size.dtype} is not a valid field size')
            
        if len(size) != self.rank:
            raise ValueError(f'size {size} is incompatible with current shape {self.shape}')

    @property
    def size(self): return self._size

    @property
    def r(self):
        'coordinates, shape = (rank, d1, d2, ..., dN)'
        return self._R

    @property
    def k(self):
        'k-space coordinates (frequencies), shape = (rank, d1, d2, ..., dN)'
        return self._K

    @property
    def dr(self):
        'coordinate spacings, shape = (rank,)'
        return self._dR

    @property
    def dk(self):
        'k-space spacings, shape = (rank,)'
        return self._dK

    @property
    def volume(self) -> NPFloat:
        'real space volume'
        return self._volume

    @property
    def dv(self) -> NPFloat:
        'real space volume element'
        return self._dV

    @property
    def R(self): 'Deprecated; use self.r instead'; return self.r

    @property
    def K(self): 'Deprecated; use self.k instead'; return self.k

    @property
    def dR(self): 'Deprecated; use self.dr instead'; return self.dr

    @property
    def dK(self): 'Deprecated; use self.dk instead'; return self.dk

    @property
    def Volume(self): 'Deprecated'; return self.volume

