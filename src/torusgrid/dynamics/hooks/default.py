from typing import TypeVar
from torusgrid.dynamics.hooks.base import EvolverHooks
from rich.progress import Progress

T = TypeVar('T')

class DefaultEvolverHooks(EvolverHooks[T]):
    def __init__(self):
        ...

    def on_multisteps_start(self, n_steps: int, n_epochs: int):
        self.pbar = Progress()
        self.task = self.pbar.add_task('multistep', total=n_epochs)

    def multisteps_enter(self):
        self.pbar.__enter__()

    def on_multisteps_step(self, step: int):
        self.pbar.advance(self.task)

    def multisteps_exit(self, *args):
        self.pbar.__exit__(*args)

    def on_multisteps_end(self): ...

    def on_nonstop_start(self, n_steps: int): ...

    def nonstop_enter(self): ...

    def on_nonstop_step(self, step: int): ...

    def nonstop_exit(self, *args): ...

    def on_nonstop_end(self): ...

    def on_interrupt(self):
        self.evolver.set_continue_flag(False)
