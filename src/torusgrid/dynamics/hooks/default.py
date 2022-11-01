from typing import TYPE_CHECKING, Optional, TypeVar
from torusgrid.dynamics.hooks.base import EvolverHooks
import time

from ...misc import console
import rich

if TYPE_CHECKING:
    from ..base import Evolver
    T = TypeVar('T', bound=Evolver)
else:
    T = TypeVar('T')

class DefaultHooks(EvolverHooks[T]):
    def on_start(self, n_steps: int, n_epochs: Optional[int]):
        self.pbar = console.make_progress_bar()
        self.task = self.pbar.add_task('', total=n_epochs, start=(n_epochs is not None))
        
    def enter_(self):
        self.pbar.__enter__()

    def exit_(self, *args):
        time.sleep(0.1)
        self.pbar.__exit__(*args)

    def on_step(self, step: int):
        self.pbar.advance(self.task)

    def pre_interrupt(self):
        rich.get_console().log('Interrupted')
    
