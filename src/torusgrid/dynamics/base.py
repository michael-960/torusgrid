from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar
from typing_extensions import Self


from .hooks.base import EvolverHooks
from .hooks.default import DefaultHooks

from ..grids import Grid
from ..fields import Field
from ..misc import context
from ..core import FFTWEffort

import threading

import time

from typing import Generic, Optional

T = TypeVar('T')

class Evolver(ABC, Generic[T]):
    """
    An Evolver handles the evolution of an object. There's no limit on
    the object's type.
    """
    def __init__(self, subject: T):
        self.subject = subject
        self._continue_flag = True

        # Backward compatibility
        self.run_nonstop = self.run
        
        self.data: Dict[str, Any] = {}
        """
        Used to store fields used by hooks, will be cleared everytime start()
        is called. I.e., before every evolution.
        """

        self.thread: threading.Thread|None = None

    @abstractmethod
    def step(self):
        """
        Run a single minimization step.
        """
        raise NotImplementedError

    def run_steps(self, n_steps: int):
        """
        Run [n_steps] minimization steps.
        """
        for i in range(n_steps):
            self.step()
            i += 1 

    def run_multisteps(self, 
            n_steps: int, n_epochs: int, 
            hooks: Optional[EvolverHooks]=None):
        """
        Run evolution loop 
        """
        hooks = self.resolve_hooks(hooks)

        self.start()
        hooks.on_multisteps_start(n_steps, n_epochs)
        with context.Context(hooks.multisteps_enter, hooks.multisteps_exit):
            for i in range(n_epochs):
                self.run_steps(n_steps)
                hooks.on_multisteps_step(i)

                if not self.get_continue_flag():
                    break
        hooks.on_multisteps_end()

    def run(self, n_steps: int,
            hooks: Optional[EvolverHooks]=None):
        """
        Run evolution loop. 
        hooks.on_nonstop_step() will be called every [n_steps] steps.
        The minimization will be run on a separate thread.
        """
        hooks = self.resolve_hooks(hooks)

        self._lock = threading.Lock()

        def run_():
            self.start()
            hooks.on_nonstop_start(n_steps)
            with context.Context(hooks.nonstop_enter, hooks.nonstop_exit):
                i = 0
                while True:
                    with self._lock:
                        if not self.get_continue_flag():
                            break
                        self.run_steps(n_steps)
                    hooks.on_nonstop_step(i)
                    i += 1

            hooks.on_nonstop_end()

        self.thread = threading.Thread(target=run_)
        self.thread.start()

        while True:
            try:
                time.sleep(1)
                if not self.get_continue_flag():
                    break
            except KeyboardInterrupt:
                hooks.pre_interrupt()
                
                self._lock.acquire()
                handled = hooks.on_interrupt()
                if self._lock.locked(): self._lock.release()

                if not handled:
                    self.set_continue_flag(False)

                hooks.post_interrupt()

        self.thread.join()

    def get_continue_flag(self) -> bool:
        return self._continue_flag

    def set_continue_flag(self, flag: bool):
        self._continue_flag = flag

    def resolve_hooks(self, hooks: Optional[EvolverHooks]) -> EvolverHooks:
        """
        Each Evolver instance is also an instance of EvolverHooks. By default
        none of the hook methods are implemented.
        """
        if hooks is None:
            _hooks = DefaultHooks()
        else:
            _hooks = hooks

        _hooks.bind(self)
        return _hooks

    def start(self) -> None:
        """
        A hook method that is called before evolution
        When overriden, super().start() should be invoked
        """
        self.data.clear()


    def wait(self):
        """
            Wait till the evolution thread ends
        """
        if self.thread is not None:
            if self._lock.locked():
                self._lock.release()
            self.thread.join()


T_grid = TypeVar('T_grid', bound=Grid)

class GridEvolver(Evolver[T_grid]):
    @property
    def grid(self) -> T_grid:
        """
        An alias for .subject
        """
        return self.subject


    def initialize_fft(
        self, *, reinit: bool = True,
        threads: int = 1,
        planning_timelimit: Optional[float]=None,
        effort: Optional[FFTWEffort] = None,
        wisdom_only: bool = False,
        destroy_input: bool = False,
        unaligned: bool = False,
    ):
        """
        Call .grid.initialize_fft() with the same arguments
        """
        fftwargs = dict(
                threads=threads, planning_timelimit=planning_timelimit,
                effort=effort, wisdom_only=wisdom_only, destroy_input=destroy_input,
                unaligned=unaligned
        )
        if reinit:
            self.grid.initialize_fft(**fftwargs)
        else:
            if not self.grid.fft_initialized():
                self.grid.initialize_fft(**fftwargs)
            else:
                raise TypeError('FFT already initialized')

T_field = TypeVar('T_field', bound=Field)

class FieldEvolver(GridEvolver[T_field]):
    @property
    def field(self) -> T_field:
        """
        An alias for .subject
        """
        return self.subject


