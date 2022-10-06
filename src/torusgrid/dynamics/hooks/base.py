from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Dict, Generic, List, Optional, TypeVar, final
from ...misc.typing import generic


if TYPE_CHECKING:
    from ..base import Evolver
    from typing_extensions import Concatenate, ParamSpec

    P = ParamSpec('P')
    T = TypeVar('T')

__hook_methods: List[str] = []

def hook(meth: Callable[Concatenate[EvolverHooks, P], None]):
    __hook_methods.append(meth.__name__)
    return meth


@generic
class EvolverHooks(Generic[T]):

    evolver: Evolver[T]

    def on_start(self, n_steps: int, n_epochs: Optional[int]): ...

    def on_step(self, step: int): ...

    def on_end(self): ...

    def enter_(self): ...

    def exit_(self, *args): ...

    @hook
    def on_multisteps_start(self, n_steps: int, n_epochs: int):
        '''
        Called before multistep evolution starts.
        self.on_start() is called by default.
        '''
        self.on_start(n_steps, n_epochs)
    
    @hook
    def multisteps_enter(self):
        self.enter_()

    @hook
    def on_multisteps_step(self, step: int):
        '''
        Called every [n_steps] during multistep evolultion. self.on_step() is
        called by default.
        '''
        self.on_step(step)


    @hook
    def multisteps_exit(self, *args):
        self.exit_(*args)

    @hook
    def on_multisteps_end(self):
        '''
        Called after multistep evolution ends.
        self.on_end() is called by default.
        '''
        self.on_end()

    @hook
    def on_nonstop_start(self, n_steps: int):
        '''
        Called before nonstop evolution starts.
        self.on_start() is called by default.
        '''
        self.on_start(n_steps, None)

    @hook
    def nonstop_enter(self):
        self.enter_()

    @hook
    def on_nonstop_step(self, step: int):
        '''
        Called every [n_steps]. self.on_step() is called by default.
        '''
        self.on_step(step)

    @hook
    def nonstop_exit(self, *args):
        self.exit_(*args)

    @hook
    def on_nonstop_end(self):
        '''
        Called after nonstop evolution ends.
        self.on_end() is called by default.
        '''
        self.on_end()

    @hook
    def on_interrupt(self):
        '''
        Called when evolver is interrupted.
        '''

    def bind(self, evolver: Evolver[T]):
        '''
        Bind evolver to hooks
        '''
        self.evolver = evolver

    def __add__(self, other: EvolverHooks):  
        return combine(self, other)



def combine(*hooks: EvolverHooks):
    '''
    Combine various hooks
    '''

    eh = EvolverHooks()

    def get_combined_func(name):
        def _combined(*args, **kwargs):
            for h in hooks:
                getattr(h, name)(*args, **kwargs)
        return _combined

    for methname in __hook_methods:
        setattr(eh, methname, get_combined_func(methname))

    return eh




