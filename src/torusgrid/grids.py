from __future__ import annotations
import threading
import tqdm
import time
import warnings
import shutil
from typing import Tuple, Union, overload

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
import pyfftw

from michael960lib.math import fourier
from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError



class ComplexGridND:
    '''A ComplexGridND object is a complex array of shape (d1, d2, .., dN)
    equipped with fourier transform. No length scales are associated with the grid. 
    '''
    def __init__(self, shape: Tuple[int, ...]):
        self.fft: Union[pyfftw.FFTW, None] = None
        self.ifft: Union[pyfftw.FFTW, None] = None
        self._isreal = False
        self.shape = shape
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
        self.fft = pyfftw.FFTW(self.psi, self.psi_k, direction='FFTW_FORWARD', axes=all_axis, **fftwargs)
        self.ifft = pyfftw.FFTW(self.psi_k, self.psi, direction='FFTW_BACKWARD', axes=all_axis, **fftwargs)
        self.set_psi(psi_tmp)
    
    def fft_initialized(self) -> bool:
        '''Whether the FFTW plans are initialized.
        '''
        return not(self.fft is None or self.ifft is None)

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

    def copy(self) -> ComplexGridND:
        '''Generate a new object with the same grid data.
        '''
        field1 = ComplexGridND(self.shape)
        field1.set_psi(self.psi)
        return field1

    def set_psi(self, psi1: Union[float, np.ndarray]) -> None:
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


class ComplexGrid1D(ComplexGridND):
    def __init__(self, N):
        super().__init__((N,))
        self.N = N

    @overload
    def set_resolution(self, N: int) -> None: ...

    @overload
    def set_resolution(self, N: Tuple[int]) -> None: ...

    @overrides(ComplexGridND)
    def set_resolution(self, N) -> None:
        '''Set the resolution (i.e. shape).
        
        Parameters: N: integer or tuple of one integer

        '''
        if type(N) is tuple:
            super().set_resolution(N)
        else:
            super().set_resolution((N,))


class RealGrid1D(RealGridND, ComplexGrid1D):
    def __init__(self, N):
        ComplexGridND.__init__(self, (N,))
        self.N = N
        self._isreal = True

    @overload
    def set_resolution(self, N: int): ...
    @overload
    def set_resolution(self, N: Tuple[int]): ...
    @overrides(RealGridND)
    def set_resolution(self, N):
        '''Set the resolution (i.e. shape).
        
        Parameters: N: integer or tuple of one integer

        '''
        if type(N) is tuple:
            super().set_resolution(X)
        else:
            super().set_resolution((X,))


class ComplexGrid2D(ComplexGridND):
    def __init__(self, Nx, Ny):
        super().__init__((Nx, Ny))
        self.Nx = Nx
        self.Ny = Ny

    @overload
    def set_resolution(self, Nx: int, Ny: int): ...
    @overload
    def set_resolution(self, shape: Tuple[int]): ...
    @overrides(ComplexGridND)
    def set_resolution(self, X, Y=None):
        '''Set the resolution (i.e. shape).
        
        Parameters: N: integer or tuple of one integer

        '''
        if type(X) is tuple:
            super().set_resolution(X)
        else:
            super().set_resolution((X, Y))

    def plot(self, lazy_factor=1, cmap='jet', vmin=-1, vmax=1):
        plt.figure(dpi=200)
        LF = lazy_factor
        ax = plt.gca()
        cm1 = ax.pcolormesh(np.real(self.psi)[::LF], cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax, orientation='horizontal', location='top', shrink=0.2)

        plt.margins(x=0, y=0, tight=True)
        plt.show()

    
class RealGrid2D(RealGridND, ComplexGrid2D):
    def __init__(self, Nx, Ny):
        ComplexGridND.__init__(self, (Nx, Ny))
        self.Nx = Nx
        self.Ny = Ny
        self._isreal = True
        
    @overload
    def set_resolution(self, Nx: int, Ny: int): ...
    @overload
    def set_resolution(self, shape: Tuple[int]): ...
    @overrides(RealGridND)
    def set_resolution(self, X, Y=None):
        if type(X) is tuple:
            super().set_resolution(X)
        else:
            super().set_resolution((X, Y))


def load_grid(filepath, is_complex=False):
    state = np.load(filepath)
    return import_grid(state, is_complex=False)


def import_grid(state, is_complex=False):
    psi = state['psi']
    shape = psi.shape
    if is_complex: 
        grid = ComplexGridND(shape)
        grid.set_psi(psi)
        return grid 
    else:
        grid = RealGrid2D(shape)
        grid.set_psi(psi)
        return grid


class StateFunction:
    def __init__(self):
        self._content = dict()

    def get_content(self) -> dict():
        return self._content.copy()

    def get_item(self, name: str):
        return self._content[name]

    def export(self) -> dict:
        return self._content


