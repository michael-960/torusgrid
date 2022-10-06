from __future__ import annotations


from typing import Callable, TypeVar
from torusgrid.dynamics.hooks.base import EvolverHooks
import numpy as np


T = TypeVar('T')


class EarlyStopping(EvolverHooks[T]):
    def __init__(self, 
            target: Callable[[T], float], 
            rtol: float, atol: float, patience: float, *,
            cumulative: bool=False,
            check_period: int=1 
        ):

        self.target = target

        self.cumulative = cumulative

        self.rtol = rtol
        self.atol = atol
        self.check_period = check_period

        self.patience = patience

        self._clock = 0

        self._value_prev: float|None = None

        self._badness = 0


    def on_step(self, step: int):
        if step % self.check_period == 0:
            self._clock = 0

            value = self.target(self.evolver.subject)

            if self._value_prev is not None:
                if np.isclose(value, self._value_prev):
                    self._badness += 1
                else:
                    if not self.cumulative: self._badness = 0

            if self._badness >= self.patience:
                self.evolver.set_continue_flag(False)

