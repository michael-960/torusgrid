from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, TypeVar

from ..base import GridEvolver

from ...core import FloatLike
from ...grids import Grid
import numpy.typing as npt

from .temporal import TemporalEvolver


T = TypeVar('T', bound=Grid)

class SecondOrderRK4(TemporalEvolver[T], GridEvolver[T]):
    """
    Absract base class for second order RK4 evolver (i.e. with momentum)
    Performs RK4 on a Grid object to solve a PDE that is second-order in time
    
    subclasses must implement:
        - psi_dot()
    """
    def __init__(self, grid: T, dt: FloatLike):
        super().__init__(grid, dt)
    
        self.grid_tmp = self.grid.copy()
        self.grid_tmp.initialize_fft()

        self.dgrid = self.grid.copy()
        self.dgrid.zero_()
        self.dgrid.initialize_fft()

        self.dgrid_tmp = self.dgrid.copy()
        self.dgrid_tmp.initialize_fft()


    @abstractmethod
    def psi_dot(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Return the first and second derivatives of psi w.r.t. time
        The derivative should be computed with self.grid_tmp, not self.grid
        """
        raise NotImplementedError
                                  
    def step(self):
        self.set_age(self.age + self.dt)

        self.grid_tmp.psi[:] = self.grid.psi
        self.dgrid_tmp.psi[:] = self.dgrid.psi

        k1, l1 = self.psi_dot()
        self.grid_tmp.psi[...] += k1*self.dt/2
        self.dgrid_tmp.psi[...] += l1*self.dt/2

        k2, l2 = self.psi_dot()
        self.grid_tmp.psi[...] += k2*self.dt/2
        self.dgrid_tmp.psi[...] += l2*self.dt/2

        k3, l3 = self.psi_dot()
        self.grid_tmp.psi[...] += k3*self.dt
        self.dgrid_tmp.psi[...] += l3*self.dt

        k4, l4 = self.psi_dot()

        self.grid.psi[...] += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.dgrid.psi[...] += self.dt / 6 * (l1 + 2*l2 + 2*l3 + l4)



class FirstOrderRK4(TemporalEvolver[T], GridEvolver):
    """
    Absract base class for first order RK4 evolver (i.e. without momentum)
    Performs RK4 on a Grid object to solve a PDE that is first-order in time
    
    subclasses must implement:
        - psi_dot()
    """

    def __init__(self, grid: T, dt: float):
        super().__init__(grid, dt)
    
        self.grid_tmp = self.grid.copy()
        self.grid_tmp.initialize_fft()

        if not self.grid.fft_initialized():
            self.grid.initialize_fft()



    @abstractmethod
    def psi_dot(self) -> npt.NDArray:
        """
        Return the first derivative of psi w.r.t. time
        The derivative should be computed with self.grid_tmp, not self.grid
        """
        raise NotImplementedError
                                  
    def step(self):
        self.age += self.dt

        self.grid_tmp.psi[...] = self.grid.psi

        k1= self.psi_dot()
        self.grid_tmp.psi[...] += k1*self.dt/2

        k2= self.psi_dot()
        self.grid_tmp.psi[...] += k2*self.dt/2

        k3= self.psi_dot()
        self.grid_tmp.psi[...] += k3*self.dt

        k4= self.psi_dot()

        self.grid.psi[...] += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)



