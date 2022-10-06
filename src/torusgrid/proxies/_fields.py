from __future__ import annotations
from typing import TYPE_CHECKING, Dict, cast
import numpy as np

from ..fields import RealField2D

from michael960lib.common import scalarize

if TYPE_CHECKING:
    from zlug.file.proxies import ObjectProxy


class RealField2DNPZ:
    @staticmethod
    def read(path: str, **kwargs) -> RealField2D:
        with open(path, 'rb') as f:
            dat = np.load(f, allow_pickle=False)

            state = dict()
            for key in dat.files:
                state[key] = dat[key]

            state = cast(Dict, scalarize(state))

            psi = state['psi']
            Lx = state['Lx']
            Ly = state['Ly']
            Nx = psi.shape[0]
            Ny = psi.shape[1]

            field = RealField2D(Lx, Ly, Nx, Ny)
            field.set_psi(psi)
        return field

    @staticmethod
    def write(path: str, data: RealField2D, **kwargs) -> None:
        assert isinstance(data, RealField2D)
        with open(path, 'wb') as f:
            np.savez(f,
                psi=data.psi.copy(),
                Lx=data.Lx,
                Ly=data.Ly
            )



