from __future__ import annotations

from typing import TypeVar
from ...core import FloatLike
from ..base import Evolver


T = TypeVar('T')


class TemporalEvolver(Evolver[T]):
    """
    For evolvers associated with time steps
    """
    def __init__(self, subject: T, dt: FloatLike):
        super().__init__(subject)
        self.dt = dt
        self.set_age(0)

    def set_age(self, age: FloatLike):
        self.age = age
        self.data['age'] = age

    def start(self) -> None:
        super().start()
        self.data['dt'] = self.dt


