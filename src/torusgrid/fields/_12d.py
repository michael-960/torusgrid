from __future__ import annotations
from typing import Tuple, overload

import numpy as np
from matplotlib import pyplot as plt

from michael960lib.common import overrides

from ..grids import ComplexGridND
from ._base import ComplexFieldND, RealFieldND


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
    def set_size(self, size: Tuple[int, int]): ...
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
    def set_dimensions(self, size: Tuple[int, int], shape: Tuple[int, int]): ...
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



