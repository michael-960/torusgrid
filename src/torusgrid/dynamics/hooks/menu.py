from __future__ import annotations


from typing import TYPE_CHECKING, Callable
import rich
from rich.live import Live

from torusgrid.dynamics.hooks.base import EvolverHooks

if TYPE_CHECKING:
    from torusgrid.dynamics.base import Evolver


class Menu(EvolverHooks):
    def __init__(self, 
            prompt: str, 
            action: Callable[[str, Evolver], None]
        ) -> None:
        self.prompt = prompt
        self.action = action

    def on_interrupt(self):

        #rich.get_console().log('Interrupted')
        live: Live = self.evolver.data['__live__']
        live.console.log('Interrupted')

        s = live.console.input(self.prompt)
        self.action(s, self.evolver)
        return True
