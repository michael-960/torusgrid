import numpy as np
from ._base import ComplexGridND, RealGridND



def load_grid(filepath: str, is_complex=False):
    state = np.load(filepath)
    return import_grid(state, is_complex=False)


def import_grid(state, is_complex=False):
    psi = state['psi']
    shape = psi.shape
    if is_complex: 
        grid = ComplexGridND(shape)
        grid.set_psi(psi)
        return grid 
    else:
        grid = RealGridND(shape)
        grid.set_psi(psi)
        return grid



class StateFunction:
    def __init__(self):
        self._content = dict()

    def get_content(self) -> dict:
        return self._content.copy()

    def get_item(self, name: str):
        return self._content[name]

    def export(self) -> dict:
        return self._content


