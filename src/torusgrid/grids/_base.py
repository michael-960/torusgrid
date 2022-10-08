from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import pyfftw


class Grid(ABC):
    _fft: Union[pyfftw.FFTW, None] = None
    _ifft: Union[pyfftw.FFTW, None] = None

    psi: npt.NDArray[np.floating] | npt.NDArray[np.complexfloating]
    psi_k: npt.NDArray[np.complexfloating]
    _fft_axes: Tuple[int, ...]
    
    def initialize_fft(self, **fftwargs) -> None:
        '''Initialize the FFTW forward and backward plans. By default the
        fourier transform plans are not initialized as it can take a
        considerable amount of time.

        Parameters:
            **fftwargs: keyword arguments to be passed to pyfftw.FFTW()

        '''
        psi_tmp = self.psi.copy()
        self._fft = pyfftw.FFTW(
                self.psi, self.psi_k,
                direction='FFTW_FORWARD',
                axes=self._fft_axes, **fftwargs)

        self._ifft = pyfftw.FFTW(
                self.psi_k, self.psi,
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
    
    @property
    def shape(self): return self.psi.shape

    @property
    def rank(self): return len(self.shape)

    @abstractmethod
    def set_psi(
            self, 
            psi1: Union[complex,
                    npt.NDArray[np.complexfloating],
                    npt.NDArray[np.floating]]
        ) -> None: ...



