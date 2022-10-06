from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, TypeVar
from .hooks import EvolverHooks, DefaultEvolverHooks
from ..misc import context
from ..misc.typing import generic

import threading

import time

T = TypeVar('T')

from typing import Generic, Optional



@generic
class Evolver(ABC, Generic[T], EvolverHooks[T]):
    '''
    An Evolver handles the evolution of an object. There's no limit on
    the object's type.
    '''
    def __init__(self, subject: T):
        self.subject = subject

        self.started = False
        self.ended = False
        self.name = 'null'
        self._continue_flag = True


        # Backward compatibility
        self.run_nonstop = self.run

    @abstractmethod
    def step(self):
        '''
        Run a single minimization step.
        '''
        raise NotImplementedError

    def run_steps(self, n_steps: int):
        '''
        Run [n_steps] minimization steps.
        '''
        for i in range(n_steps):
            self.step()
            i += 1 

    def run_multisteps(self, 
            n_steps: int, n_epochs: int, 
            hooks: Optional[EvolverHooks]=None):
        '''
        Run minimization loop 
        '''
        hooks = self.resolve_hooks(hooks)

        hooks.on_multisteps_start(n_steps, n_epochs)
        with context.Context(hooks.multisteps_enter, hooks.multisteps_exit):
            for i in range(n_epochs):
                self.run_steps(n_steps)
                hooks.on_multisteps_step(i)

                if not self.get_continue_flag():
                    break
        hooks.on_multisteps_end()

    def run(self, n_steps: int, hooks: Optional[EvolverHooks]=None):
        '''
        Run minimization loop. 
        hooks.on_nonstop_step() will be called every [n_steps] steps.
        The minimization will be run on a separate thread.
        '''
        hooks = self.resolve_hooks(hooks)

        lock = threading.Lock()
        def run():
            hooks.on_nonstop_start(n_steps)
            with context.Context(hooks.nonstop_enter, hooks.nonstop_exit):
                i = 0
                while True:
                    with lock:
                        if not self.get_continue_flag():
                            break
                        self.run_steps(n_steps)
                    hooks.on_nonstop_step(i)
                    i += 1

            hooks.on_nonstop_end()

        thread = threading.Thread(target=run)
        thread.start()

        while True:
            try:
                time.sleep(1)
                if not self.get_continue_flag():
                    break
            except KeyboardInterrupt:
                with lock:
                    hooks.on_interrupt()
        thread.join()

    def get_continue_flag(self) -> bool:
        return self._continue_flag

    def set_continue_flag(self, flag: bool):
        self._continue_flag = flag

    def resolve_hooks(self, hooks: Optional[EvolverHooks]) -> EvolverHooks:
        '''
        Each Evolver instance is also an instance of EvolverHooks. By default
        none of the hook methods are implemented.
        '''
        if hooks is None:
            return self + DefaultEvolverHooks()
        return self + hooks

