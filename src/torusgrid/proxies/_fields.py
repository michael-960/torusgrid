from __future__ import annotations
import numpy as np
from warnings import warn
from ..fields import RealField2D, ComplexField
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




class ComplexFieldNPZ:
    @staticmethod
    def read(path: str, **kwargs) -> ComplexField:
        with open(path, 'rb') as f:
            dat = np.load(f, **kwargs)

            psi = dat['psi']
            size = dat['size']
            fft_axes = tuple(dat['fft_axes'].tolist())

            precision = core.FloatingPointPrecision.from_dtype(psi.dtype)

            shape = psi.shape
            field = ComplexField(
                size,
                shape, 
                fft_axes=fft_axes,
                precision=precision
            )

            if np.all(np.isreal(psi)):
                warn('Building ComplexField object with real data', np.ComplexWarning)

            field.psi[...] = psi
        return field 

    @staticmethod
    def write(path: str, data: ComplexField, **kwargs) -> None:

        fft_axes = np.array(data.fft_axes, dtype=np.int_)
        with open(path, 'wb') as f:
            np.savez(f, 
                     psi=data.psi, 
                     size=data.size,
                     fft_axes=fft_axes)







