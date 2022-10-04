from __future__ import annotations
from typing_extensions import Self
import warnings
import shutil
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import pyfftw

from michael960lib.common import overrides


class ComplexGridND:
    '''A ComplexGridND object is a complex array of shape (d1, d2, .., dN)
    equipped with fourier transform. No length scales are associated with the grid. 
    '''
    def __init__(self, shape: Tuple[int, ...]):
        self._fft: Union[pyfftw.FFTW, None] = None
        self._ifft: Union[pyfftw.FFTW, None] = None

        self._isreal = False
        self.shape = shape
        
        self.psi: npt.NDArray[np.complex128]
        self.psi_k: npt.NDArray[np.complex128]

        self.set_resolution(shape)

    def set_resolution(self, shape: Tuple[int]) -> None:
        '''Set the resolution (i.e. shape).
        
        Parameters:
            shape: tuple of integers (d1, d2, ..., dN)

        '''
        self.shape = shape
        self.rank = len(shape)

        self.psi = pyfftw.zeros_aligned(shape, dtype='complex128')
        self.psi_k = pyfftw.zeros_aligned(shape, dtype='complex128')


    def initialize_fft(self, **fftwargs) -> None:
        '''Initialize the FFTW forward and backward plans. By default the
        fourier transform plans are not initialized as it can take a
        considerable amount of time.

        Parameters:
            **fftwargs: keyword arguments to be passed to pyfftw.FFTW()

        '''
        psi_tmp = self.psi.copy()
        all_axis = tuple(np.arange(self.rank))
        self._fft = pyfftw.FFTW(self.psi, self.psi_k, direction='FFTW_FORWARD', axes=all_axis, **fftwargs)
        self._ifft = pyfftw.FFTW(self.psi_k, self.psi, direction='FFTW_BACKWARD', axes=all_axis, **fftwargs)
        self.set_psi(psi_tmp)

    def fft(self):
        if self._fft is None:
            raise RuntimeError('FFT is not initizlized')
        self._fft()

    def ifft(self):
        if self._ifft is None:
            raise RuntimeError('IFFT is not initialized')
        self._ifft()
    
    def fft_initialized(self) -> bool:
        '''Whether the FFTW plans are initialized.
        '''
        return not (self._fft is None or self._ifft is None)

    def export_state(self) -> dict:
        '''Export the grid state to a dictionary
        '''
        state = {'psi': self.psi.copy()}
        return state

    def save(self, fname: str, verbose=False) -> None:
        '''Save the grid data into a file.

        Parameters:
            fname: the base file name. A .grid extension will be appended.
            verbose: whether to print out details
        '''
        tmp_name = f'{fname}.tmp.file'
        if verbose:
            self.yell(f'dumping profile data to {fname}.grid')
        np.savez(tmp_name, **self.export_state()) 
        shutil.move(f'{tmp_name}.npz', f'{fname}.grid')

    def copy(self) -> Self:
        '''Generate a new object with the same grid data.
        '''
        field1 = ComplexGridND(self.shape)
        field1.set_psi(self.psi)
        return field1

    def set_psi(self, psi1: Union[float, npt.NDArray[np.complex_], npt.NDArray[np.float_]]) -> None:
        '''Set grid data.

        Parameters: psi1: new grid data, can be either scalar (float) or
        np.ndarray. If a scalar is given, all entries are set to the given value.
        '''
        if not np.isscalar(psi1):
            assert isinstance(psi1, np.ndarray)
            if psi1.shape != self.shape:
                raise ValueError(f'array has incompatible shape {psi1.shape} with {self.shape}')
        self.psi[...] = psi1

    def yell(self, s):
        '''Print text with prefix [grid]
        '''
        print(f'[grid] {s}')


class RealGridND(ComplexGridND):
    '''A RealGridND object is a real array of shape (d1, d2, .., dN) equipped
    with fourier transform. No length scales are associated with the grid. '''

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__(shape)
        self._isreal = True

    def set_resolution(self, shape: Tuple[int, ...]) -> None:
        '''Set the resolution (i.e. shape).
        
        Parameters: shape: tuple of integers (d1, d2, ..., dN), last dimension
        must be even (if not it will be made even automatically)

        '''
        if shape[-1] % 2:
            warnings.warn('odd resolution on the last axis will be automatically made even for RFFT')
            shape_list = list(shape)
            shape_list[-1] += 1
            shape = tuple(shape_list)

        self.shape = shape
        self.rank = len(self.shape)

        shape_k = list(shape)
        shape_k[-1] = shape_k[-1]//2 + 1
        shape_k = tuple(shape_k)

        self.psi = pyfftw.zeros_aligned(shape, dtype='float64')
        self.psi_k = pyfftw.zeros_aligned(shape_k, dtype='complex128')
    
    @overrides(ComplexGridND)
    def set_psi(self, psi1: Union[float, np.ndarray]):
        if not np.isscalar(psi1):
            assert isinstance(psi1, np.ndarray)
            if psi1.shape != self.shape:
                raise ValueError(f'array has incompatible shape {psi1.shape} with {self.shape}')
        
        if not np.all(np.isreal(psi1)):
            raise ValueError(f'array is complex') 
        self.psi[...] = psi1

