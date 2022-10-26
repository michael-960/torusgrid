from __future__ import annotations
import numpy as np
from ..fields import RealField2D
from .. import core


class RealField2DNPZ:
    """
    RealField2D to NPZ
    """
    @staticmethod
    def read(path: str, **kwargs) -> RealField2D:
        with open(path, 'rb') as f:
            dat = np.load(f, allow_pickle=False, **kwargs)

            state = dict()
            for key in dat.files:
                state[key] = dat[key]

            state = core.scalarize(state)

            psi = state['psi']
            Lx = state['Lx']
            Ly = state['Ly']
            Nx = psi.shape[0]
            Ny = psi.shape[1]

            field = RealField2D(Lx, Ly, Nx, Ny)
            field.psi[...] = psi
        return field

    @staticmethod
    def write(path: str, data: RealField2D, **kwargs) -> None:
        assert isinstance(data, RealField2D)
        with open(path, 'wb') as f:
            np.savez(f,
                psi=data.psi.copy(),
                Lx=data.lx,
                Ly=data.ly, **kwargs
            )

