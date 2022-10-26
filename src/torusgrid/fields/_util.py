raise ImportError('Module is outdated and deprecated')
from __future__ import annotations
from typing import Literal, overload

import numpy as np

from ..core import scalarize

from ._2d import ComplexField2D, RealField2D


@overload
def load_field(filepath: str) -> RealField2D: ...

@overload
def load_field(filepath: str, is_complex: Literal[False]) -> ComplexField2D: ...

@overload
def load_field(filepath: str, is_complex: Literal[True]) -> RealField2D: ...

@overload
def load_field(filepath: str, is_complex: bool) -> RealField2D | ComplexField2D: ...

def load_field(filepath: str, is_complex: bool=False) -> RealField2D | ComplexField2D:
    state = np.load(filepath, allow_pickle=True)
    
    state1 = dict()
    for key in state.files:
        state1[key] = state[key]
    state1 = scalarize(state1)

    return import_field(state1, is_complex=is_complex)


@overload
def import_field(state: dict) -> RealField2D: ...

@overload
def import_field(state: dict, is_complex: Literal[False]) -> RealField2D: ...

@overload
def import_field(state: dict, is_complex: Literal[True]) -> ComplexField2D: ...

@overload
def import_field(state: dict, is_complex: bool) -> ComplexField2D | RealField2D: ...

def import_field(state: dict, is_complex: bool=False) -> ComplexField2D | RealField2D:
    psi = state['psi']
    Lx = state['Lx']
    Ly = state['Ly']
    Nx = psi.shape[0]
    Ny = psi.shape[1]
    if is_complex: 
        field = ComplexField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi)
        return field
    else:
        field = RealField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi)
        return field


class FieldOperationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message



