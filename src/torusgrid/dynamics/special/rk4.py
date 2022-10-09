from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple, TypeVar

from ...grids import ComplexGrid
import numpy.typing as npt

from .temporal import TemporalEvolver


T_co = TypeVar('T_co', bound=ComplexGrid, covariant=True)

class SecondOrderRK4(TemporalEvolver[T_co]):
    '''
    Performs RK4 on a ComplexGrid object to solve a PDE that is
    second-order in time
    '''
    def __init__(self, grid: T_co, dt: float):
        super().__init__(grid, dt)
    
        self.grid = self.subject
        '''
        Here self.grid == self.subject
        '''
        
        self.grid_tmp = self.grid.copy()
        self.grid_tmp.initialize_fft()

        self.dgrid = self.grid.copy()
        self.dgrid.set_psi(0)
        self.dgrid.initialize_fft()

        self.dgrid_tmp = self.dgrid.copy()
        self.dgrid_tmp.initialize_fft()


    @abstractmethod
    def psi_dot(self) -> Tuple[npt.NDArray, npt.NDArray]:
        '''
        Return the first and second derivatives of psi w.r.t. time
        The derivative should be computed with self.grid_tmp, not self.grid
        '''
        raise NotImplementedError
                                  
    def step(self):
      self.set_age(self.age + self.dt)

      self.grid_tmp.psi[:] = self.grid.psi
      self.dgrid_tmp.psi[:] = self.dgrid.psi
                              
      k1, l1 = self.psi_dot()
      self.grid_tmp.psi += k1*self.dt/2
      self.dgrid_tmp.psi += l1*self.dt/2
                              
      k2, l2 = self.psi_dot()
      self.grid_tmp.psi += k2*self.dt/2
      self.dgrid_tmp.psi += l2*self.dt/2
                              
      k3, l3 = self.psi_dot()
      self.grid_tmp.psi += k3*self.dt
      self.dgrid_tmp.psi += l3*self.dt
                              
      k4, l4 = self.psi_dot()
                              
      self.grid.psi += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
      self.dgrid.psi += self.dt / 6 * (l1 + 2*l2 + 2*l3 + l4)



class FirstOrderRK4(TemporalEvolver[T_co]):
    '''
    Performs RK4 on a ComplexGrid object to solve a PDE that is
    first-order in time
    '''
    def __init__(self, grid: T_co, dt: float):
        super().__init__(grid, dt)
    
        self.grid = self.subject
        '''
        Here self.grid == self.subject
        '''
        self.grid_tmp = self.grid.copy()
        self.grid_tmp.initialize_fft()

        if not self.grid.fft_initialized():
            self.grid.initialize_fft()



    @abstractmethod
    def psi_dot(self) -> npt.NDArray:
        '''
        Return the first derivative of psi w.r.t. time
        The derivative should be computed with self.grid_tmp, not self.grid
        '''
        raise NotImplementedError
                                  
    def step(self):
      self.age += self.dt

      self.grid_tmp.psi[:] = self.grid.psi
                              
      k1= self.psi_dot()
      self.grid_tmp.psi += k1*self.dt/2
                              
      k2= self.psi_dot()
      self.grid_tmp.psi += k2*self.dt/2
                              
      k3= self.psi_dot()
      self.grid_tmp.psi += k3*self.dt
                              
      k4= self.psi_dot()
                              
      self.grid.psi += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)







