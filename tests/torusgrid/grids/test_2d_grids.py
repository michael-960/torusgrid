from __future__ import annotations
from typing import Tuple
import pytest
from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
from torusgrid.grids import ComplexGrid2D, RealGrid2D
from scipy.fft import fft2, ifft2, rfft2, irfft2
import numpy as np

import torusgrid.misc.testutils as T
from functools import partial

_cfg = T.get_config()
config = _cfg['grids']['2d']
gconfig = _cfg['global']

ARRAY_GEN_CFG = dict(
    min_offset=config['min_offset'], max_offset=config['max_offset'],
    min_scale=config['min_scale'], max_scale=config['max_scale']
)

def get_fft_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['fft_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['fft_tol']))

shapes = T.gen_shapes(config['num_samples'], ranks=[2], 
                      min_numel=config['min_numel'],
                      max_numel=config['max_numel'])
nxs = [shape[0] for shape in shapes]
nys = [shape[1] for shape in shapes]


reals = [['real','complex'][np.random.randint(0,2)] for _ in shapes]
fft_axeses = [[None,(0,1),(0,),(1,),(1,0)][np.random.randint(0,5)] for _ in shapes]

precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]

argvals = dict(nx=nxs, ny=nys, real_=reals, fft_axes=fft_axeses, precision=precisions)


class TestGrid2D:

    @T.parametrize('nx', 'ny', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_props(
        self, 
        nx: int, ny: int, 
        fft_axes: Tuple[int,...]|None, real_: str,
        precision: PrecisionStr
    ):

        if fft_axes is None:
            fft_axes = (0,1)

        last_fft_axis = fft_axes[-1]

        real = real_ == 'real'
        shape = (nx, ny)
      
        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes, precision=precision)
        else:
            if shape[last_fft_axis] % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealGrid2D(*shape, fft_axes=fft_axes, precision=precision)
                assert g.shape[last_fft_axis] == shape[last_fft_axis] + 1
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)
            else:
                g = RealGrid2D(*shape, fft_axes=fft_axes, precision=precision)

        assert g.rank == 2
        assert last_fft_axis in [0,1]
        assert g._fft is None
        assert g._ifft is None

        assert g.last_fft_axis == fft_axes[-1]
           
        assert g.nx == shape[0]
        assert g.ny == shape[1]
        assert g.shape == shape
        assert g.psi.shape == shape


        if not real:
            shape_k = list(shape)
        else:
            shape_k = list(shape)
            shape_k[last_fft_axis] //= 2
            shape_k[last_fft_axis] += 1

        assert g.psi_k.shape == tuple(shape_k)
        assert g.shape_k == tuple(shape_k)
        assert g.psi.dtype == T.real_dtype(precision) if real else T.complex_dtype(precision)
        assert g.psi_k.dtype == T.complex_dtype(precision)
        assert g.precision is FloatingPointPrecision.cast(precision)
        assert g.fft_axes == fft_axes


    @T.parametrize('nx', 'ny', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_copy(
        self, 
        nx: int, ny: int, 
        fft_axes: Tuple[int,...]|None, real_: str,
        precision: PrecisionStr
    ):

        real = real_ == 'real'
        shape = (nx, ny) 

        if fft_axes is None:
            fft_axes = (0,1)

        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes, precision=precision)
        else:
            if shape[fft_axes[-1]] % 2 == 1:
                shape = T.even_shape(shape, fft_axes)
            g = RealGrid2D(*shape, fft_axes=fft_axes, precision=precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real), precision=precision)

        h = g.copy()
        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k
        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)
        assert T.same_meta(g, h)

        assert g.nx == h.nx
        assert g.ny == h.ny


    @pytest.mark.slow
    @T.parametrize('nx', 'ny', 'fft_axes', 'real_', 'precision', argvals=argvals)
    def test_fft(
        self, 
        nx: int, ny: int, 
        fft_axes: Tuple[int,...]|None, real_: str,
        precision: PrecisionStr
    ):
        real = real_ == 'real'
        shape = (nx, ny) 
        if fft_axes is None:
            fft_axes = (0,1)

        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes, precision=precision)
        else:
            if shape[fft_axes[-1]] % 2 == 1:
                shape = T.even_shape(shape, fft_axes)
            g = RealGrid2D(*shape, fft_axes=fft_axes, precision=precision)

        tol = get_fft_tol(g.precision)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real), precision=precision)

            g.psi[...] = f
            assert np.array_equal(g.psi, f)

            g.fft() 
            fk = fft2(f, axes=fft_axes)
            if not real:
                assert T.adaptive_allclose(g.psi_k, fk, **tol) # type: ignore
            else:
                fk_r = rfft2(f, axes=fft_axes)
                assert T.adaptive_allclose(g.psi_k, fk_r, **tol) # type: ignore

            g.ifft()
            f = ifft2(fk, axes=fft_axes)
            assert T.adaptive_allclose(g.psi, f, **tol) # type: ignore

            if real:
                fr = irfft2(fk_r, axes=fft_axes) # type: ignore
                assert T.adaptive_allclose(g.psi, fr, **tol) # type: ignore

