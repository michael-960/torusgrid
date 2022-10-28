from functools import partial
from typing import Tuple
from torusgrid import grids
from scipy.fft import fftn, ifftn
import numpy as np
import pytest
import time
from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
import torusgrid.misc.testutils as T

_cfg = T.get_config()
config = _cfg['grids']['complex']
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



class TestComplexGrid:
    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_props(self, 
                   shape: Tuple[int,...], 
                   fft_axes: Tuple[int,...],
                   precision: PrecisionStr
                   ):
        """
        Test grid properties
        """
        rank = len(shape)
        g = grids.ComplexGrid(shape, fft_axes=fft_axes, precision=precision)

        f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)

        g.psi[...] = f

        assert np.array_equal(g.psi, f)

        assert g.rank == rank
        assert g.rank == len(shape)
        assert g.numel == np.prod(shape)
        assert g.numel == np.size(f)

        assert g.shape == shape
        assert g.shape_k == shape
        assert g.fft_axes == fft_axes
        assert g.last_fft_axis == fft_axes[-1]

        assert np.all(np.iscomplex(g.psi))
        assert g.psi_k.dtype.kind == 'c'
        assert g.psi_k.dtype == T.complex_dtype(precision)
        assert g.psi.dtype == T.complex_dtype(precision)
        assert g.precision is FloatingPointPrecision.cast(precision)

        assert g._fft is None
        assert g._ifft is None

        assert not g.isreal


    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_copy(self, shape: Tuple[int, ...], fft_axes: Tuple[int, ...], precision: PrecisionStr):
        """
        Test the copy() function
        """
        g = grids.ComplexGrid(shape, fft_axes=fft_axes, precision=precision)
        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)
        h = g.copy()

        assert np.array_equal(g.psi, h.psi)
        assert g.psi is not h.psi
        assert g.psi_k is not h.psi_k

        assert T.same_meta(g, h)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)
        assert not np.array_equal(g.psi, h.psi)


    @pytest.mark.slow
    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_fft(self, shape: Tuple[int, ...], fft_axes: Tuple[int,...], precision: PrecisionStr):
        """
        Test whether fft() and ifft() yield the same results as scipy's fftn() and
        ifftn()
        """
        g = grids.ComplexGrid(shape, fft_axes=fft_axes, precision=precision)

        tol = get_fft_tol(g.precision)
        
        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)
            g.psi[...] = f

            assert np.array_equal(g.psi, f)

            g.fft()
            fk = fftn(f, axes=fft_axes)

            assert T.adaptive_allclose(
                    g.psi_k, fk,  # type: ignore
                    **tol
                    )

            g.ifft()
            f = ifftn(fk, axes=fft_axes)
            assert T.adaptive_allclose(
                    g.psi, f, # type: ignore
                    **tol)

        g.psi[...] = 0
        g.fft()
        assert np.allclose(g.psi_k, 0)


    @pytest.mark.expect
    @pytest.mark.slow
    @T.parametrize('shape', 'fft_axes', 'precision', argvals=argvals)
    def test_time(self, shape: Tuple[int,...], fft_axes: Tuple[int,...], precision: PrecisionStr):
        """
        Assert that pyfftw is faster than scipy
        """
        g = grids.ComplexGrid(shape, fft_axes=fft_axes, precision=precision)

        if g.numel > 1e4:
            threads = 4
        elif g.numel > 512:
            threads = 2
        else:
            threads = 1

        g.initialize_fft(threads=threads) 

        f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True, precision=precision)

        g.psi[...] = f

        start = time.time()
        for _ in range(config['num_fft_speed_trials']):
            fftn(f, axes=fft_axes, workers=threads, )

        end = time.time()


        t_scipy = end - start

        start = time.time()
        for _ in range(config['num_fft_speed_trials']):
            g.fft()
        end = time.time()

        t_tg = end - start
        
        assert t_tg < t_scipy
