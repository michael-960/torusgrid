from __future__ import annotations
import sys


from typing import TYPE_CHECKING, Callable, TypeVar
import rich
from rich.live import Live

from .base import EvolverHooks

if TYPE_CHECKING:
    from ..base import Evolver
    T = TypeVar('T', bound=Evolver)
else:
    T = TypeVar('T')



class ExitOnInterrupt(EvolverHooks[T]):
    def on_interrupt(self):
        live: Live = self.evolver.data['__live__']
        live.console.log('Interrupted')
        self.evolver.set_continue_flag(False)
        self.evolver.wait()
        sys.exit(0)


class Menu(EvolverHooks[T]):
    def __init__(self, 
            prompt: str, 
            action: Callable[[str, T], None]
        ) -> None:
        self.prompt = prompt
        self.action = action

    def on_interrupt(self):
        live: Live = self.evolver.data['__live__']
        live.console.log('Interrupted')

        s = live.console.input(self.prompt)
        self.action(s, self.evolver)
        return True
