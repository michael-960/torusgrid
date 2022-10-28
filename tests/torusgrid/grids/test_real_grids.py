from functools import partial
from typing import List, Tuple

from torusgrid import grids
from scipy.fft import rfftn, irfftn, fftn, ifftn
import numpy as np
import pytest
from torusgrid.core.dtypes import PrecisionLike, PrecisionStr


import torusgrid.misc.testutils as T

_cfg = T.get_config()
config = _cfg['grids']['real']
gconfig = _cfg['global']
ARRAY_GEN_CFG = dict(
    min_offset=config['min_offset'], max_offset=config['max_offset'],
    min_scale=config['min_scale'], max_scale=config['max_scale']
)

def get_fft_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['fft_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['fft_tol']))


shapes = T.gen_shapes(config['num_samples'],
                       config['ranks'],
                       min_numel=config['min_numel'],
                       max_numel=config['max_numel'])

fft_axess = [T.gen_fft_axes(len(shape)) for shape in shapes]

precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]


argvals = dict(shape=shapes, fft_axes=fft_axess, precision=precisions)


class TestRealGrids:

    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_props(self, shape: Tuple[int,...], fft_axes: Tuple[int,...], precision: PrecisionStr):
        """
        Test grid properties
        """
        rank = len(shape)

        last_fft_axis = fft_axes[-1]

        if shape[fft_axes[-1]] % 2 == 1:
            with pytest.warns(UserWarning):
                g = grids.RealGrid(shape, fft_axes=fft_axes, precision=precision)
            shape = T.even_shape(shape, fft_axes)
        else:
            g = grids.RealGrid(shape, fft_axes=fft_axes, precision=precision)

        shape_k_ = list(shape)
        shape_k_[last_fft_axis] //= 2
        shape_k_[last_fft_axis] += 1
        shape_k = tuple(shape_k_)

        f = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

        g.psi[...] = f

        assert g.rank == rank
        assert g.rank == len(shape)
        assert g.last_fft_axis == last_fft_axis
        assert g.shape[last_fft_axis] % 2 == 0
        assert g.shape == shape

        assert g.psi_k.shape == shape_k
        assert g.shape_k == shape_k

        assert g.psi.dtype == T.real_dtype(precision)
        assert g.psi_k.dtype == T.complex_dtype(precision)
        
        assert g.psi.dtype.kind == 'f'
        assert g.psi_k.dtype.kind == 'c'
        assert g.numel == T.prod(shape)
        assert g.numel == np.size(f)

        assert np.all(np.isreal(g.psi))
        assert g._fft is None
        assert g._ifft is None

        assert g.isreal

    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_copy(self, shape: Tuple[int,...], fft_axes: Tuple[int,...], precision: PrecisionStr):
        """
        Test the copy() function
        """
        if shape[fft_axes[-1]] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)
        g = grids.RealGrid(shape, fft_axes=fft_axes, precision=precision)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)

        h = g.copy()

        assert np.array_equal(g.psi, h.psi)
        assert g.psi is not h.psi
        assert g.psi_k is not h.psi_k

        assert T.same_meta(g, h)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)
        assert not np.array_equal(g.psi, h.psi)

    @pytest.mark.slow
    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_fft(self, shape: Tuple[int,...], fft_axes: Tuple[int,...], precision: PrecisionStr):
        """
        Test whether fft() and ifft() yield the same results as scipy's fftn() and
        ifftn()
        """
        if shape[fft_axes[-1]] % 2 == 1:
            shape = T.even_shape(shape, fft_axes)
        g = grids.RealGrid(shape, fft_axes=fft_axes, precision=precision)

        tol = get_fft_tol(g.precision)
        
        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft()

        for _ in range(config['num_fft_trials']):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, precision=precision)
            g.psi[...] = f

            assert np.array_equal(g.psi, f)

            g.fft()
            fk = rfftn(f, axes=fft_axes)
            fk_alt = fftn(f, axes=fft_axes)

            assert T.adaptive_allclose(g.psi_k, fk, **tol) # type: ignore

            g.ifft()
            f = irfftn(fk, axes=fft_axes)
            f_alt = ifftn(fk_alt, axes=fft_axes)

            assert T.adaptive_allclose(g.psi, f, **tol) # type: ignore
            assert T.adaptive_allclose(g.psi, f_alt, **tol) # type: ignore
            
            g.psi[tuple([0]*g.rank)] *= (1 + tol['rtol1'] * 50)
            assert not T.adaptive_allclose(g.psi, f_alt, **tol) # type: ignore

        g.psi[...] = 0
        g.fft()
        assert np.allclose(g.psi_k, 0)


