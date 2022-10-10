from __future__ import annotations


from typing import TYPE_CHECKING, TypeVar

from torusgrid.typing.general import FloatLike
from ..base import Evolver


T = TypeVar('T')

# if TYPE_CHECKING:
#     from torusgrid.dynamics.base import Evolver
# else:
#     from torusgrid.dynamics.base import Evolver as _Evolver
#     Evolver = {T: _Evolver}


class TemporalEvolver(Evolver[T]):
    '''
    For evolvers associated with time steps
    '''
    def __init__(self, subject: T, dt: FloatLike):
        super().__init__(subject)
        self.dt = dt
        self.set_age(0)

    def set_age(self, age: FloatLike):
        self.age = age
        self.data['age'] = age


