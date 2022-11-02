from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from typing import TYPE_CHECKING, Generic, Literal, Optional, Tuple, TypeVar, TypedDict, Union, overload
from warnings import warn

import numpy as np
import numpy.typing as npt

import pyfftw
from ..core import FloatLike, PrecisionLike, FloatingPointPrecision, get_real_dtype, FFTWEffort


if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar('T', np.complexfloating, np.floating)



# @generic
class Grid(ABC, Generic[T]):
    """
    Base class for grids.
    The generic type T refers to the dtype of the wrapped numpy array (self.psi).
    """
    
    _fft: Union[pyfftw.FFTW, None] = None
    _ifft: Union[pyfftw.FFTW, None] = None

    _psi: npt.NDArray[T]
    _psi_k: npt.NDArray[np.complexfloating]

    _precision: FloatingPointPrecision
    _fft_axes: Tuple[int, ...]

    def __init__(self, 
            shape: Tuple[int, ...], /, *,
            precision: PrecisionLike='double',
            fft_axes: Optional[Tuple[int,...]]=None
        ):

        self._precision = FloatingPointPrecision.cast(precision)

        if fft_axes is None:
            self._fft_axes = tuple(i for i in range(len(shape)))
        else:
            self._fft_axes = tuple(int(a) for a in fft_axes)

        self._init_grid_data(shape) # init psi and psi_k
    
    @abstractmethod
    def _init_grid_data(self, shape: Tuple[int,...]) -> None:
        """
        Initialize psi and psi_k
        """

    @abstractproperty
    def isreal(self) -> bool:
        """
        Return whether the field or grid object holds real data
        """
        ...

    def set_psi(self, psi1: npt.NDArray[T]|FloatLike) -> None:
        """
        Deprecated: use psi[...] = ... instead
        """
        raise RuntimeError('use psi[...] = ... to assign grid data instead')

    def initialize_fft(
        self, *,
        threads: int = 1,
        planning_timelimit: Optional[float]=None,
        effort: Optional[FFTWEffort] = None,
        wisdom_only: bool = False,
        destroy_input: bool = False,
        unaligned: bool = False,
    ) -> None:
        """
        Initialize the FFTW forward and backward plans. By default the
        fourier transform plans are not initialized as it can take a
        considerable amount of time.

        Parameters:
            effort: FFTW planning effort

            wisdom_only: Raise an error if no plan for the current transforms is present

            destroy_input: Whether to allow input to be destroyed during
                           transform, note that for certain transforms the
                           input is destroyed anyway

            unaligned: Alignment of the data will not be assumed.

        """

        flags = tuple( 
                      ([effort] if effort is not None else [])
                      + (['FFTW_WISDOM_ONLY'] if wisdom_only else [])
                      + (['FFTW_DESTROY_INPUT'] if destroy_input else [])
                      + (['FFTW_UNALIGNED'] if unaligned else [])
                      )

        
        psi_tmp = self.psi.copy()

        self._fft = pyfftw.FFTW(
                self._psi, self._psi_k,
                direction='FFTW_FORWARD',
                axes=self._fft_axes,
                threads=threads, planning_timelimit=planning_timelimit,
                flags=flags)

        self._ifft = pyfftw.FFTW(
                self._psi_k, self._psi,
                direction='FFTW_BACKWARD',
                axes=self._fft_axes,
                threads=threads, planning_timelimit=planning_timelimit,
                flags=flags)

        self.psi[...] = psi_tmp

    def fft(self):
        """
        Run forward FFT on psi and store results to psi_k
        If FFT plans are not initialized, a TypeError will be raised
        """
        self._fft() # type: ignore

    def ifft(self):
        """
        Run backward FFT on psi_k and store results to psi
        If FFT plans are not initialized, a TypeError will be raised
        """
        self._ifft() # type: ignore
    
    def fft_initialized(self) -> bool:
        """
        Return whether the FFTW plans are initialized.
        """
        return not (self._fft is None or self._ifft is None)
        
    @property
    def shape(self) -> Tuple[int,...]: 
        """
        Return the shape in real space (x-space / time domain etc)
        """
        return self.psi.shape

    @property
    def shape_k(self) -> Tuple[int,...]:
        """
        Return the shape in Fourier (k-space / frequency domain etc)
        """
        return self.psi_k.shape

    @property
    def rank(self): 
        """
        Return the number of axes.
        Same as numpy's ndim.
        """
        return len(self.shape)

    @property
    def numel(self) -> int: 
        """
        Return the number of elements.
        Same as numpy's size() or torch's numel()
        """
        return np.prod(self.shape)
    
    @property
    def fft_axes(self):
        """
        Return the axes along which FFT is performed
        """
        return self._fft_axes

    @property
    def last_fft_axis(self): 
        """
        Return the last FFT axis.
        Useful when dealing with RFFTs.
        """
        return self._fft_axes[-1]

    @property
    def psi(self):
        """
        Real space data
        To assign values, use .psi[...] = ..., not .psi = ...
        """
        return self._psi

    @property
    def psi_k(self): 
        """
        k-space data
        To assign values, use .psi_k[...] = ..., not .psi_k = ...
        """
        return self._psi_k

    @property
    def precision(self):
        """
        Return the floating point precision
        """
        return self._precision

    def copy(self) -> Self:
        """
        Build a new object with the same grid data.
        Must be overriden if __init__ signature is overriden
        """
        grid1 = self.__class__(
                    self.shape,
                    precision=self.precision,
                    fft_axes=self.fft_axes
                )
        grid1.psi[...] = self.psi
        return grid1

    @overload
    @classmethod
    def from_array(
        cls, x: np.ndarray, /, *, 
        metadata: dict
    ) -> Self: ...

    @overload
    @classmethod
    def from_array(
        cls, x: np.ndarray, /, *,
        precision: PrecisionLike='double',
        fft_axes: Optional[Tuple[int, ...]] = None
    ) -> Self: ...

    @classmethod
    def from_array(cls, x: np.ndarray, /, **kwargs): 
        badargs = False
        if 'metadata' in kwargs.keys():
            metadata = kwargs['metadata']
            if len(kwargs.keys()) != 1: badargs = True
            precision = metadata['precision']
            fft_axes = metadata['fft_axes']
            shape = metadata['shape']
            if x.shape != tuple(shape):
                raise ValueError(
                        f'Metadata shape {metadata["shape"]} ' +
                        f'inconsistent with array shape {x.shape}'
                    )
        else:
            if len(kwargs.keys()) != 2: badargs = True
            precision = kwargs['precision']
            fft_axes = kwargs['fft_axes']

        if badargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs}')

        g = cls(x.shape,
                precision=FloatingPointPrecision.cast(precision),
                fft_axes=fft_axes)

        g.psi[...] = x
        return g

    def metadata(self) -> dict:
        """
        Everything except the data itself
        """
        return dict(precision=self.precision.name,
                    shape=self.shape,
                    fft_axes=self.fft_axes)

    @classmethod
    def concat(cls, *metas: dict, axis: int) -> dict:
        """
        Concatenate metadata dicts.
        """
        precisions = [meta['precision'] for meta in metas]
        precision = FloatingPointPrecision.most_precise(*precisions)
        shapes = [meta['shape'] for meta in metas]
    
        fft_axes = [meta['fft_axes'] for meta in metas]

        if not fft_axes[1:] == fft_axes[:-1]:
            warn(f'Not all FFT axes are the same: {fft_axes}, the first will be used.')


        flag = False
        shape = []
    
        if not all([len(shape) == len(shapes[0]) for shape in shapes]):
            flag = True

        for ax, s in enumerate(zip(*shapes)):
            if ax != axis:
                shape.append(s[0])
                if not all([si == s[0] for si in s]):
                    flag = True
                    break
            else:
                shape.append(sum(s))

        if flag:
            raise ValueError(f'Shapes not concatenatable: {shapes}')

        return dict(
            precision=precision.name,
            shape=tuple(shape),
            fft_axes=metas[0]['fft_axes']
        )

    @classmethod
    def transpose(cls, meta: dict, axes: Tuple[int,...]) -> dict:
        shape = meta['shape']
        fft_axes = meta['fft_axes']
        newmeta = meta.copy()

        new_old_ax_map = axes
        old_new_ax_map = [
            {old:new for new,old in enumerate(new_old_ax_map)}[old_]
            for old_ in range(len(axes))
        ]

        newmeta['shape'] = tuple([shape[ax] for ax in axes])
        newmeta['fft_axes'] = tuple([old_new_ax_map[fft_ax] for fft_ax in fft_axes])
        return newmeta

    @classmethod
    def crop(cls, meta: dict, axis: int, a: int, b: int) -> dict:
        shape = meta['shape']
        if not 0 <= a < b < shape[axis]:
            raise ValueError(f'Invalid cropping range: shape={shape}, axis={axis}, a={a}, b={b}')

        newmeta = meta.copy()
        newmeta['shape'] = tuple([(b - a) if ax == axis else shape[ax]
                                  for ax in range(len(shape))])

        return newmeta

    def zero_(self):
        """
        Set all elements of psi to zero
        psi_k will not be affected
        """
        self.psi[...] = get_real_dtype(self.precision)('0.')


