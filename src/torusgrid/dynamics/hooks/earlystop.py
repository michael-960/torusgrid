from __future__ import annotations
from abc import abstractmethod
from textwrap import dedent

from typing import TYPE_CHECKING, Callable, Literal, Optional, TypeVar

import rich
from rich.console import Group, RenderableType
from rich.panel import Panel


from .base import EvolverHooks
from ...core import FloatLike


import numpy as np

from torusgrid.misc.console import make_progress_bar

if TYPE_CHECKING:
    from ..base import Evolver
    T = TypeVar('T', bound=Evolver)
else:
    T = TypeVar('T')


class EarlyStopping(EvolverHooks[T]):
    """
    Stops the evolver when self.condition() returns true enough times
    """
    def __init__(self, 
            patience: int, *,
            cumulative: bool=False,
            period: int=1,
            verbose: bool=True,
            show_progress: bool=True
        ):

        self.cumulative = cumulative
        self.check_period = period
        self.patience = patience
        self._badness = 0

        self.verbose = verbose
        self.show_progress = show_progress

    def get_status(self) -> RenderableType:
        return dedent(f'''\
            Monitoring: {self.get_monitor_target()}
            Status: {self._badness}/{self.patience}'''
        )

    def on_start(self, n_steps: int, n_epochs: Optional[int]):
        self._badness = 0

    def enter_(self):
        if self.show_progress:
            self.panel: Panel = self.evolver.data['__panel__']

            self.pbar = make_progress_bar(text=f'Monitoring {self.get_monitor_target()}', 
                    transient=True)

            self.task = self.pbar.add_task(self.get_monitor_target(), total=self.patience)

            self.panel.renderable = Group(self.panel.renderable, self.pbar)


    def on_step(self, step: int):
        if step % self.check_period == 0:
            if self.condition():

                if self.show_progress:
                    self.pbar.advance(self.task)

                self._badness += 1
            else:
                if not self.cumulative:
                    self._badness = 0
                    self.pbar.update(self.task, completed=0)

            if self._badness >= self.patience:
                if self.verbose:
                    rich.get_console().log('Critierion met, stopping...')
                self.evolver.set_continue_flag(False)

    @abstractmethod
    def condition(self) -> bool: ...


    @abstractmethod
    def get_monitor_target(self) -> str: ...


class DetectSlow(EarlyStopping[T]):
    """
        Monitor a value (via target), and stop the evolver when the value
        changes sufficiently slowly.

        target can be either a function (T) -> float or a string. If a string
        is provided, evolve.data[target] will be used.
    """
    def __init__(self, 
            target: Callable[[T], float] | str,
            rtol: FloatLike, atol: FloatLike,
            patience: int, *, 
            monotone: Literal['increase', 'decrease', 'ignore'] = 'ignore',
            cumulative: bool = False,
            period: int = 1, 
            verbose: bool = True,
            show_progress: bool = True
            ):
        
        super().__init__(patience, 
                cumulative=cumulative,
                period=period,
                verbose=verbose,
                show_progress=show_progress)

        self.rtol = rtol
        self.atol = atol

        self.target = target

        self._val_prev: None|float = None

        self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol)

        if monotone == 'increase':
            self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol) and x >= y

        if monotone == 'decrease':
            self.compare = lambda x, y: np.isclose(x, y, rtol=rtol, atol=atol) and x <= y

        self.verbose = verbose

        self._target = str(self.target)

    def enter_(self):
        super().enter_()

    def get_monitor_target(self):
        return f'{self._target} rtol={self.rtol} atol={self.atol}'

    def on_start(self, n_steps: int, n_epochs: Optional[int]):
        self._badness = 0
        self._val_prev = None

    def condition(self) -> bool:
        if isinstance(self.target, str):
            val: float = self.evolver.data[self.target]
        else:
            val = self.target(self.evolver.subject)

        cond = False

        if self._val_prev is not None:
            cond = self.compare(val, self._val_prev)

        self._val_prev = val

        return cond



