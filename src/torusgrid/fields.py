import threading
import tqdm
import time
import warnings
from typing import List, Tuple, overload
import shutil

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
import pyfftw

from michael960lib.math import fourier
from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError
from michael960lib.common import deprecated, experimental, scalarize
from .grids import ComplexGrid2D, RealGrid2D, ComplexGridND, RealGridND



class ComplexFieldND(ComplexGridND):
    def __init__(self, size: Tuple[float], shape: Tuple[int]):
        super().__init__(shape)
        self.set_size(size)

    def set_dimensions(self, size: Tuple[float], shape: Tuple[int]):
        self.set_resolution(shape)
        self.set_size(size)

    def set_size(self, size: Tuple[int]):
        if len(size) != self.rank:
            raise ValueError(f'size {size} is incompatible with current shape {self.shape}')
        self.size = size
        self.Volume = np.prod(size)
        
        R, K, DR, DK = [], [], [], []

        for i in range(self.rank):
            x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i]) 
            R.append(x)
            K.append(k)
            DR.append(dx)
            DK.append(dk)

        self.R = np.meshgrid(*R, indexing='ij')
        self.K = np.meshgrid(*K, indexing='ij')
        self.dR = np.array(DR)
        self.dK = np.array(DK)

        self.dV = np.prod(self.dR)

    @overrides(ComplexGridND)
    def export_state(self) -> dict():
        state = {'psi': self.psi.copy(), 'size': self.size}
        return state

    @overrides(ComplexGridND)
    def copy(self):
        field1 = self.__class__(self.size, self.shape)
        field1.set_psi(self.psi)
        return field1

    @overrides(ComplexGrid2D)
    def save(self, fname: str, verbose=False):
        tmp_name = f'{fname}.tmp.file'
        if verbose:
            self.yell(f'dumping field data to {fname}.field')
        np.savez(tmp_name, **self.export_state()) 
        shutil.move(f'{tmp_name}.npz', f'{fname}.field')


class RealFieldND(ComplexFieldND, RealGridND):
    def __init__(self, size: Tuple[float], shape: Tuple[int]):
        super().__init__(size, shape)
        self._isreal = True

    @overrides(ComplexFieldND)
    def set_size(self, size: Tuple[int]):
        if len(size) != self.rank:
            raise ValueError(f'size {size} is incompatible with current shape {self.shape}')
        self.size = size
        self.Volume = np.prod(size)
        
        R, K, DR, DK = [], [], [], []

        for i in range(self.rank):
            if i == self.rank - 1:
                x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i], real=True)
            else:
                x, k, dx, dk = fourier.generate_xk(size[i], self.shape[i]) 

            R.append(x)
            K.append(k)
            DR.append(dx)
            DK.append(dk)

        self.R = np.meshgrid(*R, indexing='ij')
        self.K = np.meshgrid(*K, indexing='ij')
        self.dR = np.array(DR)
        self.dK = np.array(DK)

        self.dV = np.prod(self.dR)


class ComplexField2D(ComplexFieldND):
    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int):
        super().__init__((Lx, Ly), (Nx, Ny))
        self.set_dimensions((Lx, Ly), (Nx, Ny))

    def get_dimensions(self):
        return self.Lx, self.Ly, self.Nx, self.Ny


    @overload
    def set_dimensions(self, size, shape): ...
    @overload
    def set_dimensions(self, Lx, Ly, Nx, Ny): ...
    @overrides(ComplexFieldND)
    def set_dimensions(self, *args):
        if len(args) == 2:
            if type(args[0]) is tuple and type(args[1]) is tuple:
                if len(args[0]) == len(args[1]) == 2:
                    super().set_dimensions(args[0], args[1])
                else:
                    raise ValueError(f'incompatible size/shape for 2D: {args[0]}/{args[1]}')
            else:
                raise ValueError(f'invalid size/shape')
        elif len(args) == 4:
            super().set_dimensions((args[0], args[1]), (args[2], args[3]))
        else:
            raise ValueError
           

    @overload
    def set_size(self, Lx: float, Ly: float): ...
    @overload
    def set_size(self, size: Tuple[int]): ...
    @overrides(ComplexFieldND)
    def set_size(self, arg1, arg2=None):
        if type(arg1) is tuple and arg2 is None:
            if len(arg1) == 2:
                super().set_size(arg1)
            else:
                raise ValueError(f'incompatible shape for 2D size: {arg1}')
        else:
            super().set_size((Lx, Ly))

        self.setup_convenience_variables()

    def setup_convenience_variables(self):

        self.Lx = self.size[0]
        self.Ly = self.size[1]
        self.Nx = self.shape[0]
        self.Ny = self.shape[1]

        self.dx = self.dR[0]
        self.dy = self.dR[1]

        self.dkx = self.dK[0]
        self.dky = self.dK[1]
 
        self.X = self.R[0]
        self.Y = self.R[1]

        #self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')
        self.Kx = self.K[0]
        self.Ky = self.K[1]

        self.Kx2 = self.Kx**2
        self.Ky2 = self.Ky**2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2

        self.K2 = self.Kx2 + self.Ky2
        self.K4 = self.K2**2
        self.K6 = self.K2 * self.K4

    @overrides(ComplexFieldND)
    def export_state(self):
        state = super().export_state()
        state['Lx'] = self.Lx
        state['Ly'] = self.Ly
        return state
    
    @overrides(ComplexFieldND)
    def copy(self):
        field1 = self.__class__(self.Lx, self.Ly, self.Nx, self.Ny)
        field1.set_psi(self.psi)
        return field1

    def plot(self, lazy_factor=1, cmap='jet', vmin=-1, vmax=1, show=True):
        plt.figure(dpi=200)
        LF = lazy_factor
        ax = plt.gca()
        cm1 = ax.pcolormesh(self.X[::LF], self.Y[::LF], np.real(self.psi)[::LF], cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax, orientation='horizontal', location='top', shrink=0.2)

        plt.margins(x=0, y=0, tight=True)
        if show:
            plt.show()
        else:
            return ax


class RealField2D(RealFieldND, ComplexField2D):
    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int):
        ComplexGridND.__init__(self, (Nx, Ny))
        self.set_dimensions((Lx, Ly), (Nx, Ny))
        self._isreal = True

    @overload
    def set_dimensions(self, size: Tuple[int], shape: Tuple[int]): ...
    @overload
    def set_dimensions(self, Lx: float, Ly: float, Nx: int, Ny: int): ...
    @overrides(RealFieldND)
    def set_dimensions(self, *args):
        ComplexField2D.set_dimensions(self, *args)

    @overload
    def set_size(self, Lx: float, Ly: float): ...
    @overload
    def set_size(self, size: Tuple[int]): ...
    @overrides(RealFieldND)
    def set_size(self, arg1, arg2=None):
        if type(arg1) is tuple and arg2 is None:
            if len(arg1) == 2:
                RealFieldND.set_size(self, arg1)
            else:
                raise ValueError(f'incompatible shape for 2D size: {arg1}')
        else:
            RealFieldND.set_size(self, (arg1, arg2))

        self.setup_convenience_variables()

    @overrides(RealFieldND)
    def export_state(self) -> dict:
        return ComplexField2D.export_state(self)

    @overrides(RealFieldND)
    def copy(self):
        return ComplexField2D.copy(self)


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
        field.set_psi(psi)
        return field
    else:
        field = RealField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi)
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



