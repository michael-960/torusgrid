import threading
import tqdm
import time
import warnings
from typing import List
import shutil

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
import pyfftw

from michael960lib.math import fourier
from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError
from michael960lib.common import deprecated, experimental, scalarize
from .grids import ComplexGrid2D, RealGrid2D



@deprecated('will be removed')
def real_convolution_2d(psi_k, kernel, NN):
    r = kernel * (np.abs(psi_k**2))
    r[:,0] /= 2
    r[:,-1] /= 2
    return np.sum(r) / (NN/2)


# A Field2D is a complex 2D field with definite dimensions, it also includes:
# psi, psi_k(fourier transform), forward plan, backward plan
class ComplexField2D(ComplexGrid2D):
    def __init__(self, Lx, Ly, Nx, Ny):
        super().__init__(Nx, Ny)
        self.set_dimensions(Lx, Ly, Nx, Ny)
        self.fft2 = None
        self.ifft2 = None
        
    def set_dimensions(self, Lx, Ly, Nx, Ny):
        self.set_resolution(Nx, Ny)
        self.set_size(Lx, Ly)

    def set_size(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        self.Volume = Lx*Ly

        x, kx, self.dx, self.dkx, y, ky, self.dy, self.dky = fourier.generate_xk_2d(Lx, Ly, self.Nx, self.Ny, real=self._isreal)
        self.dV = self.dx * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')

        self.Kx2 = self.Kx**2
        self.Ky2 = self.Ky**2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2

        self.K2 = self.Kx2 + self.Ky2
        self.K4 = self.K2**2
        self.K6 = self.K2 * self.K4

    @overrides(ComplexGrid2D)
    def export_state(self):
        state = {'psi': self.psi.copy(), 'Lx': self.Lx, 'Ly': self.Ly}
        return state
    
    @overrides(ComplexGrid2D)
    def copy(self):
        field1 = self.__class__(self.Lx, self.Ly, self.Nx, self.Ny)
        field1.set_psi(self.psi)
        return field1

    @overrides(ComplexGrid2D)
    def plot(self, lazy_factor=1, cmap='jet', vmin=-1, vmax=1):
        plt.figure(dpi=200)
        LF = lazy_factor
        ax = plt.gca()
        cm1 = ax.pcolormesh(self.X[::LF], self.Y[::LF], np.real(self.psi)[::LF], cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax, orientation='horizontal', location='top', shrink=0.2)

        plt.margins(x=0, y=0, tight=True)
        plt.show()

    @overrides(ComplexGrid2D)
    def save(self, fname: str, verbose=False):
        tmp_name = f'{fname}.tmp.file'
        if verbose:
            self.yell(f'dumping profile data to {fname}.field')
        np.savez(tmp_name, **self.export_state()) 
        shutil.move(f'{tmp_name}.npz', f'{fname}.field')

    @overrides(ComplexGrid2D)
    def yell(self, s):
        print(f'[field] {s}')


class RealField2D(RealGrid2D, ComplexField2D):
    def __init__(self, Lx, Ly, Nx, Ny):
        RealGrid2D.__init__(self, Nx, Ny)
        self.set_dimensions(Lx, Ly, Nx, Ny)
        self.fft2 = None
        self.ifft2 = None


def load_field(filepath, is_complex=False) -> ComplexField2D:
    state = np.load(filepath, allow_pickle=True)
    
    state1 = dict()
    for key in state.files:
        state1[key] = state[key]
    state1 = scalarize(state1)

    return import_field(state1, is_complex=False)

def import_field(state, is_complex=False) -> RealField2D:
    psi = state['psi']
    Lx = state['Lx']
    Ly = state['Ly']
    Nx = psi.shape[0]
    Ny = psi.shape[1]
    if is_complex: 
        field = ComplexField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi, verbose=False)
        return field
    else:
        field = RealField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi, verbose=False)
        return field


class FieldStateFunction:
    def __init__(self):
        self._content = dict()

    def get_content(self) -> dict():
        return self._content.copy()

    def get_item(self, name: str):
        return self._content[name]

    def export(self) -> dict:
        return self._content


class FieldOperationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message



