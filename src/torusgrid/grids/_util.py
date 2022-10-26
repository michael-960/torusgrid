raise ImportError('Module is outdated and deprecated')
import numpy as np
from ._complex import ComplexGrid
from ._real import RealGrid



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





