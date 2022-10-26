from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, TypeVar, final

import numpy as np
import numpy.typing as npt

from ..core import get_real_dtype, PrecisionStr, NPFloat, SizeLike

from ..grids import Grid

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar('T', np.complexfloating, np.floating)

class Field(Grid[T]):
    """
    Base class for fields.
    """
    _size: npt.NDArray[np.floating]
    _R: npt.NDArray[np.floating]
    _dR: npt.NDArray[np.floating]

    _K: npt.NDArray[np.floating]
    _K2: npt.NDArray[np.floating]
    _K4: npt.NDArray[np.floating]
    _K6: npt.NDArray[np.floating]
    _dK: npt.NDArray[np.floating]

    _volume: NPFloat
    _dV: NPFloat

    def __init__(self,
            size: Sequence[np.floating|float] | npt.NDArray[np.floating],
            shape: Tuple[int, ...], *,
            precision: PrecisionStr = 'double',
            fft_axes: Optional[Tuple[int,...]] = None
        ):

        "call Grid contructor"
        Grid[T].__init__(
            self, shape,
            precision=precision, 
            fft_axes=fft_axes
        )
        self._init_coordinate_vars()
        self.set_size(size)

    @abstractmethod
    def _update_coordinate_vars(self) -> None:
        """
        Given size, update _R, _K, _dR, _dK
        """

    @final
    def _init_coordinate_vars(self):
        """
        Called by the Field constructure to initialize _size, _R, _K, _dR, _dK
        arrays to ZEROS
        """
        dtype = get_real_dtype(self._precision)
        self._size = np.zeros((self.rank,), dtype=dtype)
        self._R = np.zeros((self.rank, *self.shape), dtype=dtype)
        self._K = np.zeros((self.rank, *self.shape_k), dtype=dtype)
        self._dR = np.zeros((self.rank,), dtype=dtype)
        self._dK = np.zeros((self.rank,), dtype=dtype)
    
    def set_size(self, size: SizeLike):
        """
        Set system size (dimension lengths) and update size, r, k, dr, dk etc
        """
        self.validate_size(size)

        self._size[...] = size
        self._update_coordinate_vars()

        self._K2 = np.sum(self._K**2, axis=0)
        self._K4 = self._K2**2
        self._K6 = self._K2**3

        self._volume = np.prod(self._size)
        self._dV = np.prod(self._dR)

    def validate_size(self, size: SizeLike):
        if isinstance(size, np.ndarray):
            if size.ndim != 1:
                raise ValueError(f'ndarray with shape {size.shape} is not a valid field size')

            if not np.all(np.isreal(size)):
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
    def k2(self):
        'k squared, shape = (d1, d2, ..., di/2+1, ... dN)'
        return self._K2

    @property
    def k4(self):
        'k^4, shape = (d1, d2, ..., di/2+1, ..., dN)'
        return self._K4

    @property
    def k6(self):
        'k^6, shape = (d1, d2, ..., di/2+1, ..., dN)'
        return self._K6

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
    def Volume(self): 'Deprecated; use self.volume instead'; return self.volume

    def copy(self) -> Self:
        field1 = self.__class__(
                self.size, self.shape,
                precision=self._precision,
                fft_axes=self._fft_axes
        )
        field1.psi[...] = self.psi
        return field1


