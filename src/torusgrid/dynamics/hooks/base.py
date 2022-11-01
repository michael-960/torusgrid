from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Dict, Generic, List, Optional, TypeVar


if TYPE_CHECKING:
    from ..base import Evolver
    from typing_extensions import Concatenate, ParamSpec
    P = ParamSpec('P')
    T = TypeVar('T', bound=Evolver)
else:
    T = TypeVar('T')

__hook_methods: Dict[str, bool] = {}

def hook(meth: Callable[Concatenate[EvolverHooks, P], None]):
    __hook_methods[meth.__name__] = False
    return meth

def reverse_hook(meth: Callable[Concatenate[EvolverHooks, P], None]):
    __hook_methods[meth.__name__] = True
    return meth



class EvolverHooks(Generic[T]):
    """
    Hooks called during evolution. The generic type T refers to the evolver
    class.
    """

    evolver: T

    def on_start(self, n_steps: int, n_epochs: Optional[int]): ...

    def on_step(self, step: int): ...

    def on_end(self): ...

    def enter_(self): ...

    def exit_(self, *args): ...

    @hook
    def on_multisteps_start(self, n_steps: int, n_epochs: int):
        """
        Called before multistep evolution starts.
        self.on_start() is called by default.
        """
        self.on_start(n_steps, n_epochs)
    
    @hook
    def multisteps_enter(self):
        self.enter_()

    @hook
    def on_multisteps_step(self, step: int):
        """
        Called every [n_steps] during multistep evolultion. self.on_step() is
        called by default.
        """
        self.on_step(step)


    @reverse_hook
    def multisteps_exit(self, *args):
        self.exit_(*args)

    @reverse_hook
    def on_multisteps_end(self):
        """
        Called after multistep evolution ends.
        self.on_end() is called by default. Call order is reversed.
        """
        self.on_end()

    @hook
    def on_nonstop_start(self, n_steps: int):
        """
        Called before nonstop evolution starts.
        self.on_start() is called by default.
        """
        self.on_start(n_steps, None)

    @hook
    def nonstop_enter(self):
        self.enter_()

    @hook
    def on_nonstop_step(self, step: int):
        """
        Called every [n_steps]. self.on_step() is called by default.
        """
        self.on_step(step)

    @reverse_hook
    def nonstop_exit(self, *args):
        """
        Call order is reversed
        """
        self.exit_(*args)

    @reverse_hook
    def on_nonstop_end(self):
        """
        Called after nonstop evolution ends.
        self.on_end() is called by default.
        Call order ir reversed.
        """
        self.on_end()

    @hook
    def pre_interrupt(self):
        """
        Called when evolver is interrupted and **before**
        the field lock is acquired
        """

    @hook
    def on_interrupt(self):
        """
        Called when evolver is interrupted and **after**
        the field lock is acquired.
        """

    @reverse_hook
    def post_interrupt(self):
        """
        Called after evolver is interrupted
        """


    @hook
    def bind(self, evolver: T):
        """
        Bind evolver to hooks
        """
        self.evolver = evolver

    def __add__(self, other: EvolverHooks):  
        return combine(self, other)



def combine(*hooks: EvolverHooks):
    """
    Combine various hooks
    """

    eh = EvolverHooks()

    def get_combined_func(name, reverse=False):
        if not reverse:
            def _combined(*args, **kwargs):
                handled = None
                for h in hooks:
                    handled_ = getattr(h, name)(*args, **kwargs)
                    handled = none_or(handled, handled_)
                return handled
            return _combined
        else:
            def _combined(*args, **kwargs):
                handled = None
                for h in hooks[::-1]:
                    handled_ = getattr(h, name)(*args, **kwargs)
                    handled = none_or(handled, handled_)
                return handled

            return _combined


    for methname, reverse in __hook_methods.items():
        setattr(eh, methname, get_combined_func(methname, reverse=reverse))

    return eh



def none_or(x: bool|None, y: bool|None):
    """
        True, True/False/None -> True
        False, False/None -> False
        None, None -> None
    """
    if x is None: return y
    if x is True: return True
    if x is False: return y is True


