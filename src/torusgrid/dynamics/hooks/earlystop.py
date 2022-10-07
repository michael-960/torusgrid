from __future__ import annotations
from abc import abstractmethod
from textwrap import dedent


from typing import Callable, Literal, TypeVar

import rich
from rich.box import MINIMAL
from rich.console import Group, RenderableType
from rich.panel import Panel
from torusgrid.dynamics.hooks.base import EvolverHooks


import numpy as np


T = TypeVar('T')


class EarlyStopping(EvolverHooks[T]):
    '''
    Stops the evolver when self.condition() returns true enough times
    '''
    def __init__(self, 
            patience: float, *,
            cumulative: bool=False,
            period: int=1,
            verbose: bool=True,
            show_status: bool=True
        ):

        self.cumulative = cumulative
        self.check_period = period
        self.patience = patience
        self._badness = 0

        self.verbose = verbose
        self.show_status = show_status

    def get_status(self) -> RenderableType:
        return dedent(f'''\
            Monitoring: {self.get_monitor_target()}
            Status: {self._badness}/{self.patience}
        ''')

    def enter_(self):
        if self.show_status:
            self.panel: Panel = self.evolver.data['__panel__']
            self.status: Panel = Panel(self.get_status(), 
                    box=MINIMAL, 
                    title='Early Stopping',
                    title_align='left')
            self.panel.renderable = Group(self.panel.renderable, self.status)

    def on_step(self, step: int):
        if step % self.check_period == 0:
            if self.condition():

                if self.show_status:
                    self.status.renderable = self.get_status()

                self._badness += 1
            else:
                if not self.cumulative: self._badness = 0

            if self._badness >= self.patience:
                if self.verbose:
                    rich.get_console().log('Stopping critierion met, quitting ...')
                
                self.evolver.set_continue_flag(False)

    @abstractmethod
    def condition(self) -> bool: ...


    @abstractmethod
    def get_monitor_target(self) -> str: ...


class DetectSlow(EarlyStopping[T]):
    def __init__(self, 
            target: Callable[[T], float] | str,
            rtol: float, atol: float,
            monotone: Literal['increase', 'decrease', 'ignore'],
            patience: float, *, 
            cumulative: bool = False,
            period: int = 1, verbose: bool = True):
        '''
        Monitor a value (via target), and stop the evolver when the value
        changes sufficiently slowly.

        target can be either a function (T) -> float or a string. If a string
        is provided, evolve.data[target] will be used.
        '''

        super().__init__(patience, cumulative=cumulative, period=period)
        self.target = target

        self._val_prev: None|float = None

        self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol)

        if monotone == 'increase':
            self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol) and x >= y

        if monotone == 'decrease':
            self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol) and x <= y

        self.verbose = verbose

        
        self._target = str(self.target)

    def get_monitor_target(self):
        return self._target

    def condition(self):
        if isinstance(self.target, str):
            val: float = self.evolver.data[self.target]
        else:
            val = self.target(self.evolver.subject)

        cond = False

        if self._val_prev is not None:
            cond = self.compare(val, self._val_prev)


        self._val_prev = val

        return cond



