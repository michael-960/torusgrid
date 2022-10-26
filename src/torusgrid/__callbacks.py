raise ImportError('torusgrid.callbacks is deprecated. Use torusgrid.dynamics.hooks instead')

from __future__ import annotations
from enum import Enum

import numpy as np

from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError

import warnings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dynamics import FancyEvolver



class EvolverCallBack:
    def __init__(self):
        pass

    def on_call(self, ev: FancyEvolver, sf: FieldStateFunction):
        raise NotImplementedError

    def stop_evolver(self, ev: FancyEvolver):
        ev.set_continue_flag(False)

    # for reusing the callback obj 
    def reinit(self):
        raise NotImplementedError


class EvolverMonitor(Enum):
    DIFFERENCE = 0
    OSCILLATION = 1
    

class EarlyStopping(EvolverCallBack):
    def __init__(self, patience: int, cumulative=False):
        self.p = 0
        self.patience = patience
        self.cumulative = cumulative

    @overrides(EvolverCallBack)
    def on_call(self, ev: FancyEvolver, sf: FieldStateFunction):
        #ev.info['callback'] = f'{self.p}/{self.patience}'

        if self.condition(ev, sf):
            self.p += 1
        else:
            if not self.cumulative:
                self.p = 0

        if self.p >= self.patience:
            self.stop_evolver(ev)

    def condition(self, ev: FancyEvolver, sf: FieldStateFunction) -> bool:
        return False

    @overrides(EvolverCallBack)
    def reinit(self):
        self.p = 0


class MonitoredEarlyStopping(EarlyStopping):
    def __init__(self, monitor: EvolverMonitor, target_key,
            patience: int, cumulative: bool=False,
            rtol: float=0, atol: float=0):
        super().__init__(patience, cumulative)
        self.monitor = monitor
        self.val = 1e16
        self.val_prev = 1e16
        self.val_prev2 = 1e16

        self.rtol = rtol
        self.atol = atol
        self.target_key = target_key

    @overrides(EarlyStopping)
    def on_call(self, ev: FancyEvolver, sf: FieldStateFunction):
        self.val_prev2 = self.val_prev
        self.val_prev = self.val 
        self.val = sf.get_item(self.target_key)
        super().on_call(ev, sf)

    @overrides(EarlyStopping)
    def condition(self, ev: FancyEvolver, sf: FieldStateFunction) -> bool:
        if self.monitor is EvolverMonitor.DIFFERENCE:
            return np.isclose(self.val, self.val_prev, self.rtol, self.atol)

        if self.monitor is EvolverMonitor.OSCILLATION:
            return (self.val - self.val_prev) * (self.val_prev - self.val_prev2) < 0

        return False

    @overrides(EvolverCallBack)
    def reinit(self):
        super().reinit()
        self.val = 1e16
        self.val_prev = 1e16
        self.val_prev2 = 1e16






