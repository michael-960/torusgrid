from __future__ import annotations
from abc import abstractmethod
from typing import List, Protocol, TypeVar, Union, final

from torusgrid.dynamics.base import Evolver
from torusgrid.typing.general import FloatLike
from .temporal import TemporalEvolver
from ...grids import Grid


T_co = TypeVar('T_co', bound=Grid, covariant=True)


class Step(Protocol):
    def __call__(self, dt: FloatLike) -> None: ...


class SplitStep(TemporalEvolver[T_co]):
    def __init__(self, grid: T_co, dt: FloatLike):
        super().__init__(grid, dt)
        self.grid = self.subject
        '''
        Here self.grid == self.subject
        '''

        self.realspace_steps = self.get_realspace_steps()
        self.kspace_steps = self.get_kspace_steps()

    @abstractmethod
    def get_realspace_steps(self) -> List[Step]:
        raise NotImplementedError

    @abstractmethod
    def get_kspace_steps(self) -> List[Step]:
        raise NotImplementedError

    def step(self):
        self.set_age(self.age + self.dt)

        for step in self.realspace_steps:
            step(self.dt/2)
     
        self.grid.fft()

        for step in self.kspace_steps[:-1]:
            step(self.dt/2)

        self.kspace_steps[-1](self.dt)

        for step in self.kspace_steps[:-1][::-1]:
            step(self.dt/2)

        self.grid.ifft()

        for step in self.realspace_steps[::-1]:
            step(self.dt/2)




