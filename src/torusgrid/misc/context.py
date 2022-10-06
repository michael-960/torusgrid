from __future__ import annotations
from types import TracebackType
from typing import ContextManager, Protocol, Type


class EnterFunc(Protocol):
  def __call__(self): ...


class ExitFunc(Protocol):
  def __call__(self, __exc_type: Type[BaseException] | None,                                
          __exc_value: BaseException | None,                                
          __traceback: TracebackType | None) -> bool | None:
      ...


class Context(ContextManager):
  def __init__(self, _enter: EnterFunc|None=None, _exit: ExitFunc|None=None):
      self._enter = _enter
      self._exit = _exit

  def __enter__(self) -> None:
      if self._enter is not None: self._enter()

  def __exit__(self, __exc_type: Type[BaseException] | None,                                
          __exc_value: BaseException | None,                                
          __traceback: TracebackType | None) -> bool | None:

      if self._exit is not None: return self._exit(__exc_type, __exc_value, __traceback)                                


