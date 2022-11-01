from __future__ import annotations
from abc import abstractmethod
from typing import List, Protocol, TypeVar


from ...core import FloatLike
from .temporal import TemporalEvolver
from ...grids import Grid
from ..base import GridEvolver


T = TypeVar('T', bound=Grid)


class Step(Protocol):
    def __call__(self, dt: FloatLike) -> None: ...


class SplitStep(TemporalEvolver[T], GridEvolver[T]):
    """
    Split-step algorithms.
    
    Subclasses must implement:
        - get_realspace_steps
        - get kspace_steps
    """
    def __init__(self, grid: T, dt: FloatLike):
        super().__init__(grid, dt)
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


