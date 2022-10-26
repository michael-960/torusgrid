from typing import List, Tuple
from torusgrid import grids
from scipy.fft import fftn, ifftn
import numpy as np
import pytest
import time

import torusgrid.misc.testutils as T

NUM_SHAPES = 200
NUM_FFT_OPERATIONS = 128
NUM_FFT_THREADS = 8

NUM_FFTSPEED_OPERATIONS = 128

FFT_RTOL = 1e-10
FFT_ATOL_FACTOR = 1e-8

MAX_OFFSET = 0
MIN_OFFSET = -1

MAX_SCALE = 1e9
MIN_SCALE = 1e-9

MIN_NUMEL = 1
MAX_NUMEL = 1e5

TOL = dict(rtol=FFT_RTOL, atol_factor=FFT_ATOL_FACTOR)
ARRAY_GEN_CFG = dict(
    min_offset=MIN_OFFSET, max_offset=MAX_OFFSET,
    min_scale=MIN_SCALE, max_scale=MAX_SCALE
)

shapes = T.gen_shapes(NUM_SHAPES, 
                      [1,1,2,2,2,2,3,3,3,4,4,5,6,7,8],
                      min_log_numel=np.log(MIN_NUMEL),
                      max_log_numel=np.log(MAX_NUMEL))

fft_axeses = [T.gen_fft_axes(len(shape)) for shape in shapes]

_test_ids = [f'shape={T.int_tup_repr(shape)} fft={T.int_tup_repr(fft_ax)}' 
             for shape,fft_ax in zip(shapes, fft_axeses)]


def parametrize(f):
    _f = pytest.mark.parametrize(
            ['shape', 'fft_axes'], zip(shapes, fft_axeses), ids=_test_ids
        )(f)
    return _f


class TestComplexGrid:
    @parametrize
    def test_props(self, shape: Tuple[int,...], fft_axes: Tuple[int,...]):
        """
        Test grid properties
        """
        rank = len(shape)
        g = grids.ComplexGrid(shape, fft_axes=fft_axes)

        f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)

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
        assert g.precision == 'DOUBLE'

        assert g._fft is None
        assert g._ifft is None

        assert not g.isreal


    @parametrize
    def test_copy(self, shape: Tuple[int, ...], fft_axes: Tuple[int, ...]):
        """
        Test the copy() function
        """
        g = grids.ComplexGrid(shape, fft_axes=fft_axes)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)

        h = g.copy()

        assert np.array_equal(g.psi, h.psi)
        assert g.psi is not h.psi
        assert g.psi_k is not h.psi_k


        assert g.rank == h.rank
        assert g.shape == h.shape
        assert g.shape_k == h.shape_k
        assert g.fft_axes == h.fft_axes
        assert g.isreal == h.isreal

        assert g.numel == h.numel


        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)
        assert not np.array_equal(g.psi, h.psi)


    @pytest.mark.slow
    @parametrize
    def test_fft(self, shape: Tuple[int, ...], fft_axes: Tuple[int,...]):
        """
        Test whether fft() and ifft() yield the same results as scipy's fftn() and
        ifftn()
        """
        g = grids.ComplexGrid(shape, fft_axes=fft_axes)
        
        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)
            g.psi[...] = f

            assert np.array_equal(g.psi, f)

            g.fft()
            fk = fftn(f, axes=fft_axes)
            assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore

            g.ifft()
            f = ifftn(fk, axes=fft_axes)
            assert T.adaptive_atol_allclose(g.psi, f, **TOL) # type: ignore

        g.psi[...] = 0
        g.fft()
        assert np.allclose(g.psi_k, 0)


    @pytest.mark.expect
    @pytest.mark.slow
    @parametrize
    def test_time(self, shape: Tuple[int,...], fft_axes: Tuple[int,...]):
        """
        Assert that pyfftw is faster than scipy
        """
        # shape, fft_axes = shape_fft_axes
        g = grids.ComplexGrid(shape, fft_axes=fft_axes)

        if g.numel > 1e4:
            threads = 4
        elif g.numel > 512:
            threads = 2
        else:
            threads = 1

        g.initialize_fft(threads=threads) 

        f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)

        g.psi[...] = f

        start = time.time()
        for _ in range(NUM_FFTSPEED_OPERATIONS):
            fftn(f, axes=fft_axes, workers=threads, )

        end = time.time()


        t_scipy = end - start

        start = time.time()
        for _ in range(NUM_FFTSPEED_OPERATIONS):
            g.fft()
        end = time.time()

        t_tg = end - start
        
        assert t_tg < t_scipy
