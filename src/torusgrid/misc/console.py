from __future__ import annotations
from rich.console import Console
from rich import get_console
from rich.style import StyleType
from rich.progress import (
        ProgressColumn, Progress, SpinnerColumn, MofNCompleteColumn,
        TimeElapsedColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn
        )
from typing import Optional, Tuple


def make_progress_bar(console: Optional[Console]=None,
          bar_width_multiplier=.5,
          back_style: StyleType='bar.back',
          complete_style: StyleType='bar.complete',
          finished_style: StyleType='bar.finished',
          pulse_style: StyleType='bar.pulse',
  
          spinner: Optional[str|Tuple[str,str]]='dots', text: Optional[str]=None,
          count: bool=True, percentage: bool=True,
          time_remaining: bool=True, time_elapsed: bool=True, 
          *extra_columns: ProgressColumn,
          transient: bool=False
          ) -> Progress:
      '''
      Return a Rich Progress object
      '''
      if console is None:
          console = get_console()
  
      columns = []
  
      if spinner: 
          if isinstance(spinner, str):
              columns.append(SpinnerColumn(spinner))
          else:
              columns.append(SpinnerColumn(spinner[0], finished_text=spinner[1]))
  
      if text:
          columns.append(text)
  
      if count:
          columns.append(MofNCompleteColumn())
  
      columns.append(BarColumn(bar_width=int(bar_width_multiplier*console.width),
          style=back_style, complete_style=complete_style, finished_style=finished_style,
          pulse_style=pulse_style
          ))
      BarColumn()
  
      if percentage:
          columns.append(TaskProgressColumn())
  
      if time_remaining:
          columns.append(TimeRemainingColumn())
  
      if time_elapsed:
          columns.append(TimeElapsedColumn())
  
  
      progress = Progress(*columns, console=console, transient=transient)
  
      return progress

