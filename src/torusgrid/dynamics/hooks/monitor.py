from __future__ import annotations


from typing import Callable, Dict, Literal, TypeVar
from torusgrid.dynamics.hooks.base import EvolverHooks



T = TypeVar('T')

class MonitorValues(EvolverHooks[T]):
    def __init__(self, 
            target: Dict[str, Callable[[T], float]] | Callable[[T], Dict[str, float]],
            *, 
            period: int=1) -> None:
        super().__init__()

        self.period = period
        self.target = target


        if isinstance(self.target, dict): 
            def record_():
                for key, func in self.target.items():
                    self.evolver.data[key] = func(self.evolver.subject)
        else:
            def record_():
                values = self.target(self.evolver.subject) # type: ignore
                self.evolver.data.update(values)

        self.record_values = record_

    def on_step(self, step: int):
        if step % self.period == 0:
            self.record_values()



