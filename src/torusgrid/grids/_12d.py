from __future__ import annotations
from typing import Tuple, overload

import numpy as np
from matplotlib import pyplot as plt

from michael960lib.common import overrides

from ._base import ComplexGridND, RealGridND


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



