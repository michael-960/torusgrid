import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
import pyfftw

from michael960lib.math import fourier
from michael960lib.common import overrides, IllegalActionError, ModifyingReadOnlyObjectError
from michael960lib.common import deprecated, experimental

import threading
import tqdm
import time
import warnings
from typing import List
import sys

from .fields import ComplexField2D, RealField2D, FieldStateFunction


class FieldEvolver:
    def __init__(self, field: ComplexField2D):
        self.field = field
        self.started = False
        self.ended = False
        self.name = 'null'

        self.callbacks = []
        self._continue_flag = True

    # update field
    def step(self):
        raise NotImplementedError

    def run_steps(self, N_steps: int):
        for i in range(N_steps):
            self.step()
            i += 1

    def run_multisteps(self, N_steps: int, N_epochs: int, 
            callbacks: List=[], **kwargs):
        self.start()
        self.callbacks = callbacks

        progress_bar = tqdm.tqdm(range(N_epochs), bar_format='{l_bar}{bar}|{postfix}')
        self.on_create_progress_bar(progress_bar)
        for i in progress_bar:
            self.run_steps(N_steps)
            self.on_epoch_end(progress_bar)

            if not self.get_continue_flag():
                break

        self.end()

    def run_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None, 
            callbacks: List=[], **kwargs):
        self.start()
        self.callbacks = callbacks

        lock = threading.Lock()
        stopped = False 
        def run():
            while True:
                with lock:
                    if stopped:
                        break
                    if not self.get_continue_flag():
                        break
                    self.run_steps(N_steps)
                self.on_nonstop_epoch_end()

        thread = threading.Thread(target=run)
        thread.start()

        while True:
            try:
                time.sleep(1)
                if not self.get_continue_flag():
                    break
            except KeyboardInterrupt as e:
                with lock:
                    if custom_keyboard_interrupt_handler is None:
                        stopped = True 
                        break
                    else:
                        time.sleep(0.1)
                        stopped = custom_keyboard_interrupt_handler(self)
                        if stopped:
                            break

        thread.join()
        print()
        self.end()

    def get_continue_flag(self):
        return self._continue_flag

    def set_continue_flag(self, flag):
        self._continue_flag = flag

    def on_create_progress_bar(self, progress_bar: tqdm.tqdm):
        pass

    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        for cb in self.callbacks:
            cb(self)

    def on_nonstop_epoch_end(self):
        for cb in self.callbacks:
            cb(self)
        time.sleep(0.001)

    def start(self):
        if self.ended:
            raise MinimizerError(self, 'minimization has already ended')
        self.started = True

    def end(self):
        if self.started:
            self.ended = True
        else:
            raise MinimizerError(self, 'minimization has not been started')


class FancyEvolver(FieldEvolver):
    def __init__(self, field: ComplexField2D):
        super().__init__(field)
        self.label = 'NULL'
        self.history = EvolverHistory()
        self.display_format = '[DISPLAY FORMAT UNSPECIFIED]'
        self.info = dict()
    
    def set_display_format(self, display_format: str):
        if display_format is not None:
            self.display_format = display_format
        
    @overrides(FieldEvolver)
    def start(self):
        super().start()
        sf = self.get_state_function()
        es = self.get_evolver_state()
        self.history.append_state_function(es, sf)

    @overrides(FieldEvolver)
    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        sf = self.get_state_function()
        es = self.get_evolver_state()
        progress_bar.set_description_str(self.display_format.format(**self.info, **es, **(sf.get_content())))
        self.history.append_state_function(es, sf)
        for cb in self.callbacks:
            cb(self, state)

    @overrides(FieldEvolver)
    def on_nonstop_epoch_end(self):
        sf = self.get_state_function()
        es = self.get_evolver_state()
        sys.stdout.write('\r' + self.display_format.format(**self.info, **es, **(sf.get_content())))
        self.history.append_state_function(es, sf)
        for cb in self.callbacks:
            cb(self, sf)

    @overrides(FieldEvolver)
    def end(self):
        super().end()
        self.history.commit(self.label, self.field)

    def get_evolver_state(self) -> dict:
        raise NotImplementedError 

    def get_state_function(self) -> FieldStateFunction:
        raise NotImplementedError


class EvolverHistory:
    def __init__(self):
        self.evolver_states = []
        self.state_functions = []
        self.committed = False 
        self.final_field_state = None

    def append_state_function(self, evolver_state: dict, sf: FieldStateFunction):
        if self.committed:
            raise ModifyingReadOnlyObjectError(
            f'history object is already committed and hence not editable', self)

        self.state_functions.append(sf)

    def commit(self, label:str, field: ComplexField2D):
        if self.committed:
            raise IllegalActionError(f'history object (label=\'{self.label}\') is already committed')

        self.final_field_state = field.export_state()
        self.label = label
        self.committed = True 

    def is_committed(self):
        return self.committed

    def get_state_functions(self) -> List[FieldStateFunction]:
        return self.state_functions

    def get_final_field_state(self):
        if not self.committed:
            raise IllegalActionError('cannot get final state from uncommitted PFC minimizer history')
        return self.final_field_state

    def get_label(self):
        return self.label

    def export(self) -> dict:
        if self.committed:
            state = dict()
            state['label'] = self.label
            state['state_functions'] = [sf.export() for sf in self.state_functions]
            return state
        else:
            raise IllegalActionError(
            'history object (label=\'{self.label}\') has not been committed and is therefore not ready to be exported')


class NoiseGenerator2D:
    @deprecated('NoiseGenerator2D is not stable and may be removed in a future version')
    def __init__(self, seed, amplitude, Nx, Ny, noise_type='gaussian'):
        self.seed = seed
        self.amplitude = amplitude
        self.Nx = Nx
        self.Ny = Ny

        if noise_type == 'gaussian':
            self.generate = lambda: np.random.normal(0, amplitude, size=(Nx, Ny))
        else:
            raise ValueError(f'{noise_error} is not a recognized noise type')

    def generate(self):
        raise NotImplementedError()


class FreeEnergyFunctional2D:
    def __init__(self):
        raise NotImplementedError()

    def free_energy_density(self, field: ComplexField2D):
        raise NotImplementedError()

    def free_energy(self, field: ComplexField2D):
        return np.sum(self.free_energy_density(field)) * field.dV

    def derivative(self, field: ComplexField2D):
        raise NotImplementedError()

    def mean_free_energy_density(self, field: ComplexField2D):
        return np.mean(self.free_energy_density(field))


class MinimizerError(IllegalActionError):
    def __init__(self, minimizer: FieldEvolver, msg=None):
        if minimizer is None:
            self.message = 'no minimizer provided'
        else:
            self.message = msg
        super().__init__(self.message)
        self.minimier = minimizer

