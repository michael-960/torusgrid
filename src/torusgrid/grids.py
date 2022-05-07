import numpy as np

from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
import pyfftw

from michael960lib.math import fourier
from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError
import threading
import tqdm
import time
import warnings



class ComplexGrid2D:
    def __init__(self, Nx, Ny):
        self.set_resolution(Nx, Ny)
        self.fft2 = None
        self.ifft2 = None
        self._isreal = False
        
    def set_resolution(self, Nx, Ny, verbose=False):
        self.Nx = Nx
        self.Ny = Ny
        self.psi = pyfftw.zeros_aligned((Nx, Ny), dtype='complex128')
        self.psi_k = pyfftw.zeros_aligned((Nx, Ny), dtype='complex128')

    def initialize_fft(self, **fftwargs):
        psi_tmp = self.psi.copy()
        self.fft2 = pyfftw.FFTW(self.psi, self.psi_k, direction='FFTW_FORWARD', axes=(0,1), **fftwargs)
        self.ifft2 = pyfftw.FFTW(self.psi_k, self.psi, direction='FFTW_BACKWARD', axes=(0,1), **fftwargs)
        self.set_psi(psi_tmp)

    def fft_initialized(self):
        return not(self.fft2 is None or self.ifft2 is None)

    def set_psi(self, psi1, verbose=False):
        if not np.isscalar(psi1):
            if (psi1.shape[0] != self.Nx or psi1.shape[1] != self.Ny): 
                raise ValueError(f'array has incompatible shape {psi1.shape} with ({self.Nx, self.Ny})')

        self.psi[:,:] = psi1
        if verbose:
            self.yell('new psi set')

    def save(self, target_npz, verbose=False):
        if verbose:
            self.yell(f'dumping profile data to {target_npz}')
        np.savez(target_npz, **self.export_state()) 

    def export_state(self):
        state = {'psi': self.psi.copy()}
        return state
    
    def copy(self):
        field1 = self.__class__(self.Nx, self.Ny)
        field1.set_psi(self.psi)
        return field1

    def plot(self, lazy_factor=1, cmap='jet', vmin=-1, vmax=1):
        plt.figure(dpi=200)
        LF = lazy_factor
        ax = plt.gca()
        cm1 = ax.pcolormesh(np.real(self.psi)[::LF], cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax, orientation='horizontal', location='top', shrink=0.2)

        plt.margins(x=0, y=0, tight=True)
        plt.show()

    def yell(self, s):
        print(f'[grid] {s}')


class RealGrid2D(ComplexGrid2D):
    def __init__(self, Nx, Ny):
        ComplexGrid2D.__init__(self, Nx, Ny)
        self._isreal = True
        
    def set_resolution(self, Nx, Ny):
        if Ny % 2:
            warnings.warn('odd Ny will be automatically made even for RFFT')
            Ny += 1
        self.Nx = Nx
        self.Ny = Ny
        self.psi = pyfftw.zeros_aligned((Nx, Ny), dtype='float64')
        self.psi_k = pyfftw.zeros_aligned((Nx, Ny//2+1), dtype='complex128')

    def set_psi(self, psi1, verbose=False):
        if not np.isscalar(psi1):
            if (psi1.shape[0] != self.Nx or psi1.shape[1] != self.Ny): 
                raise ValueError(f'array has incompatible shape {psi1.shape} with ({self.Nx, self.Ny})')
        
        if not np.all(np.isreal(psi1)):
            raise ValueError(f'array is complex') 

        self.psi[:,:] = psi1
        if verbose:
            self.yell('new psi set')


def load_grid(filepath, is_complex=False):
    state = np.load(filepath)
    return import_grid(state, is_complex=False)

def import_grid(state, is_complex=False):
    psi = state['psi']
    Nx = psi.shape[0]
    Ny = psi.shape[1]
    if is_complex: 
        grid = ComplexGrid2D(Nx, Ny)
        grid.set_psi(psi, verbose=False)
        return grid 
    else:
        grid = RealGrid2D(Nx, Ny)
        grid.set_psi(psi, verbose=False)
        return grid

