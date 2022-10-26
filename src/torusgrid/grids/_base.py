from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from typing import TYPE_CHECKING, Generic, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

import pyfftw

from ..core import PrecisionStr, FloatLike, generic

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar('T', np.complexfloating, np.floating)


@generic
class Grid(ABC, Generic[T]):
    """
    Base class for grids.
    The generic type T refers to the dtype of the wrapped numpy array (self.psi).
    """
    
    _fft: Union[pyfftw.FFTW, None] = None
    _ifft: Union[pyfftw.FFTW, None] = None

    _psi: npt.NDArray[T]
    _psi_k: npt.NDArray[np.complexfloating]

    _precision: PrecisionStr
    _fft_axes: Tuple[int, ...]

    def __init__(self, 
            shape: Tuple[int, ...], *,
            precision: PrecisionStr='double',
            fft_axes: Optional[Tuple[int,...]]=None
        ):

        self._precision = precision.upper() # type: ignore

        if fft_axes is None:
            self._fft_axes = tuple(np.arange(len(shape)))
        else:
            self._fft_axes = fft_axes

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

    def initialize_fft(self, **fftwargs) -> None:
        """
        Initialize the FFTW forward and backward plans. By default the
        fourier transform plans are not initialized as it can take a
        considerable amount of time.

        Parameters:
            **fftwargs: keyword arguments to be passed to pyfftw.FFTW()

        """
        psi_tmp = self.psi.copy()
        self._fft = pyfftw.FFTW(
                self._psi, self._psi_k,
                direction='FFTW_FORWARD',
                axes=self._fft_axes, **fftwargs)

        self._ifft = pyfftw.FFTW(
                self._psi_k, self._psi,
                direction='FFTW_BACKWARD',
                axes=self._fft_axes, **fftwargs)

        # self.set_psi(psi_tmp)
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
    def shape(self): 
        """
        Return the shape in real space (x-space / time domain etc)
        """
        return self.psi.shape

    @property
    def shape_k(self):
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
    def precision(self) -> Literal['SINGLE', 'DOUBLE', 'LONGDOUBLE']:
        """
        Return the floating point precision
        """
        return self._precision # type: ignore

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



