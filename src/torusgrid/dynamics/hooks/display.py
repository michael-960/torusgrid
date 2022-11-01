from __future__ import annotations
from typing import Any, Callable, Dict, Optional, TypeVar
import rich

from rich.align import AlignMethod

from torusgrid.dynamics.hooks.default import DefaultHooks

from .base import EvolverHooks

from ...misc import console


from rich.live import Live
from rich.panel import Panel as Panel_
from rich.text import Text as Text_
from rich.console import Group
from rich import box

T = TypeVar('T')


class Display(EvolverHooks):
    """
    Provide evolver.data with a live diplay object that can be accessed via
    evolver.data['__live__']

    """
    def __init__(self): ...
    
    def on_start(self, n_steps, n_epochs):
        self.live = Live('', console=rich.get_console())
        
        if '__live__' in self.evolver.data.keys():
            raise RuntimeError('Only one Display object is allowed')

        self.evolver.data['__live__'] = self.live
        
    def enter_(self):
        self.live.start()
        #self.live.__enter__()
        
    def exit_(self, *args):
        #self.live.__exit__(*args)
        self.live.stop()
        del self.evolver.data['__live__']
        del self.live


class Panel(EvolverHooks):
    """
    Draw a panel on __live__. A Display object must be present before applying
    a Panel instance.
    """
    def __init__(self, 
                 title: str | Callable[[Dict[str,Any]], str]='', 
                 title_align: AlignMethod='center', **kwargs) -> None:
        self.title = title
        self.title_align: AlignMethod = title_align
        self.kwargs = kwargs

    def get_title(self):
        if isinstance(self.title, str):
            return self.title.format(**self.evolver.data)
        else:
            return self.title(self.evolver.data).format(**self.evolver.data)

    def on_start(self, 
            n_steps: int, n_epochs: Optional[int]):

        self.live: Live = self.evolver.data['__live__']
        self.panel = Panel_(
                Group(), 
                title=self.get_title(),
                title_align=self.title_align,
                **self.kwargs
            )
        self.evolver.data['__panel__'] = self.panel

    def enter_(self):
        self.live.update(self.panel)

    def exit_(self, *args):
        del self.evolver.data['__panel__']
        del self.panel


class Progress(EvolverHooks):
    """
    """
    def __init__(self, pbar_text: str, *, period: int=1):
        self.pbar_text = pbar_text
        self.period = period
    
    def on_start(self, n_steps, n_epochs):
        self.panel: Panel_ = self.evolver.data['__panel__']
        self.pbar = console.make_progress_bar(text='{task.fields[info]}')
        self.task = self.pbar.add_task('', total=n_epochs, start=(n_epochs is not None), info='')
        
    def enter_(self):
        self.panel.renderable = Group(self.panel.renderable, self.pbar)
        
    def exit_(self, *args):
        ...
    
    def on_step(self, step):
        self.pbar.advance(self.task)
        if step % self.period == 0:
            self.pbar.update(self.task, info=self.pbar_text.format(**self.evolver.data))



class Text(EvolverHooks):
    """
    Display a text message on __panel__. A Panel object must be present before applying
    a Text instance.
    """
    def __init__(self, text: str | Callable[[Dict[str,Any]], str], 
                 *, period: int=1) -> None:
        self._text = text
        self.period = period

    def get_text(self) -> str:
        if isinstance(self._text, str):
            return self._text.format(**self.evolver.data)

        return self._text(self.evolver.data).format(**self.evolver.data)

    def on_start(self, n_steps: int, n_epochs: Optional[int]):
        self.panel: Panel_ = self.evolver.data['__panel__']
        self.textpanel = Panel_('', box=box.MINIMAL)
   
    def enter_(self):
        self.panel.renderable = Group(self.panel.renderable, self.textpanel)

    def on_step(self, step: int):
        if step % self.period == 0:
            self.textpanel.renderable = self.get_text()


