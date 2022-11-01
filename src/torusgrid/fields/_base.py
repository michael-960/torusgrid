from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, TypeVar, final, overload

import numpy as np
import numpy.typing as npt

from torusgrid.core.dtypes import FloatingPointPrecision

from ..core import get_real_dtype, PrecisionLike, NPFloat, SizeLike

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
            size: SizeLike,
            shape: Tuple[int, ...], /, *,
            precision: PrecisionLike = 'double',
            fft_axes: Optional[Tuple[int,...]] = None
        ):
        """
        . 
        """

        "call Grid contructor"
        Grid.__init__(
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

    def copy(self) -> Self:
        field1 = self.__class__(
            self.size, self.shape,
            precision=self._precision,
            fft_axes=self._fft_axes
        )
        field1.psi[...] = self.psi
        return field1

    @overload
    @classmethod
    def from_array(
        cls, x:np.ndarray, /, *, metadata: dict
    ) -> Self: ...

    @overload
    @classmethod
    def from_array(
        cls, 
        x: np.ndarray, /, *,
        size: SizeLike,
        precision: PrecisionLike = 'double',
        fft_axes: Optional[Tuple[int, ...]] = None
    ) -> Self: ...
    
    @classmethod
    def from_array(cls, x: np.ndarray, /, **kwargs) -> Self:
        badargs = False
        if 'metadata' in kwargs.keys():
            metadata = kwargs['metadata']
            if len(kwargs.keys()) != 1: badargs = True

            size = metadata['size']
            precision = metadata['precision']
            fft_axes = fft_axes = metadata['fft_axes']

            shape = metadata['shape']
            if x.shape != tuple(shape):
                raise ValueError(
                        f'Metadata shape {metadata["shape"]} ' +
                        f'inconsistent with array shape {x.shape}'
                    )
        else:
            if len(kwargs.keys()) != 3: badargs = True
            size = kwargs['size']
            precision = kwargs['precision']
            fft_axes = kwargs['fft_axes']

        if badargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')

        g = cls(size, 
                x.shape, 
                precision=FloatingPointPrecision.cast(precision),
                fft_axes=fft_axes)

        g.psi[...] = x
        return g

    def metadata(self) -> dict:
        res = Grid.metadata(self)
        res['size'] = self.size
        return res

    @classmethod
    def concat(cls, *metas: dict, axis: int) -> dict:
        res = Grid.concat(*metas, axis=axis)

        sizes = [meta['size'] for meta in metas]
        shapes = [meta['shape'] for meta in metas]

        size = []

        flag = False
        for ax, s in enumerate(zip(*sizes)):
            if ax != axis:
                if not all([np.isclose(s[0], si, rtol=1e-8, atol=0) for si in s]):
                    flag = True
                    break
                size.append(s[0])
            else:
                size.append(sum(s))

        if not all([
            np.isclose(
                sizes[0][axis] / shapes[0][axis],
                size[axis] / shape[axis], rtol=1e-8, atol=0)
            for size,shape in zip(sizes,shapes)
        ]):
            flag = True

        if flag:
            raise ValueError(
                f'Incompatible dimensions: \n sizes={sizes}, shapes={shapes}'
            )
        res['size'] = size 
        return res

    @classmethod
    def transpose(cls, meta: dict, axes: Tuple[int, ...]) -> dict:
        newmeta = Grid.transpose(meta, axes)
        size = meta['size']
        newsize = [size[ax] for ax in axes]
        newmeta['size'] = newsize
        return newmeta

    @classmethod        
    def crop(cls, meta: dict, axis: int, a: int, b: int) -> dict:
        oldshape = meta['shape']
        newmeta = super().crop(meta, axis, a, b)
        newsize = meta.copy()['size']
        newsize[axis] = newsize[axis] * (b-a) / oldshape[axis]
        newmeta['size'] = newsize
        return newmeta




