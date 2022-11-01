from __future__ import annotations


from typing import TYPE_CHECKING, Callable, Dict, TypeVar
from ...core.dtypes import FloatLike
from ..hooks.base import EvolverHooks


if TYPE_CHECKING:
    from ..base import Evolver
    T = TypeVar('T', bound=Evolver)
else:
    T = TypeVar('T')



class MonitorValues(EvolverHooks[T]):
    """
    Monitor values/expressions and store them into evolver's data field.
    """
    def __init__(self, 
            target: Dict[str, Callable[[T], FloatLike|None]] | 
                    Callable[[T], Dict[str, FloatLike|None]],
            *, 
            period: int=1) -> None:
        super().__init__()

        self.period = period
        self.target = target

        if isinstance(self.target, dict): 
            def record_():
                for key, func in self.target.items():
                    self.evolver.data[key] = func(self.evolver)
        else:
            def record_():
                values = self.target(self.evolver) # type: ignore
                self.evolver.data.update(values)

        self.record_values = record_

    def on_step(self, step: int):
        if step % self.period == 0:
            self.record_values()



