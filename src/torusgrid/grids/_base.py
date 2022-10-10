from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from typing import Generic, Optional, Tuple, TypeVar, Union
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import pyfftw


from ..misc.typing import generic
from ..typing import PrecisionStr, FloatLike


T = TypeVar('T', np.complexfloating, np.floating)

@generic
class Grid(ABC, Generic[T]):
    '''
    Base class for grids.
    The generic type T refers to the dtype of the wrapped numpy array (self.psi).
    '''
    
    _fft: Union[pyfftw.FFTW, None] = None
    _ifft: Union[pyfftw.FFTW, None] = None

    _psi: npt.NDArray[T]
    _psi_k: npt.NDArray[np.complexfloating]

    _precision: PrecisionStr
    _fft_axes: Tuple[int, ...]

    @abstractmethod
    def __init__(self, 
            shape: Tuple[int, ...], *,
            precision: PrecisionStr='double',
            fft_axes: Optional[Tuple[int,...]]=None
        ): ...


    @abstractproperty
    def isreal(self) -> bool: ...
    
    def initialize_fft(self, **fftwargs) -> None:
        '''Initialize the FFTW forward and backward plans. By default the
        fourier transform plans are not initialized as it can take a
        considerable amount of time.

        Parameters:
            **fftwargs: keyword arguments to be passed to pyfftw.FFTW()

        '''
        psi_tmp = self.psi.copy()
        self._fft = pyfftw.FFTW(
                self._psi, self._psi_k,
                direction='FFTW_FORWARD',
                axes=self._fft_axes, **fftwargs)

        self._ifft = pyfftw.FFTW(
                self._psi_k, self._psi,
                direction='FFTW_BACKWARD',
                axes=self._fft_axes, **fftwargs)

        self.set_psi(psi_tmp)

    def fft(self):
        if self._fft is None: raise RuntimeError('FFT is not initialized')
        self._fft()

    def ifft(self):
        if self._ifft is None: raise RuntimeError('IFFT is not initialized')
        self._ifft()
    
    def fft_initialized(self) -> bool:
        '''Whether the FFTW plans are initialized.
        '''
        return not (self._fft is None or self._ifft is None)

    @abstractmethod
    def set_psi(self, psi1: npt.NDArray[T]|FloatLike) -> None: ...
    
    @property
    def shape(self): return self.psi.shape

    @property
    def rank(self): return len(self.shape)
    
    @property
    def last_fft_axis(self): return self._fft_axes[-1]

    @property
    def psi(self): 'real space data'; return self._psi

    @property
    def psi_k(self): 'k-space data'; return self._psi_k

    def copy(self) -> Self:
        '''Generate a new object with the same grid data.
        '''
        grid1 = self.__class__(
                    self.shape,
                    precision=self._precision,
                    fft_axes=self._fft_axes
                )
        grid1.set_psi(self.psi)
        return grid1

    




