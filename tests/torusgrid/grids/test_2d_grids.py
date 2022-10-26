from __future__ import annotations
from typing import Tuple
import pytest
from torusgrid.grids import ComplexGrid2D , RealGrid2D
import torusgrid.misc.testutils as T
from scipy.fft import fft2, ifft2, rfft2, irfft2
import numpy as np


NUM_SHAPES = 200
NUM_FFT_OPERATIONS = 128
NUM_FFT_THREADS = 8

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

isreal = ['real', 'complex']

shapes = T.gen_shapes(NUM_SHAPES, ranks=[2], 
                      min_log_numel=np.log(MIN_NUMEL),
                      max_log_numel=np.log(MAX_NUMEL))

tol = dict(rtol=FFT_RTOL, atol_factor=FFT_ATOL_FACTOR)

fft_axeses = [None, (0,1), (0,), (1,), (1,0)]

class TestGrid2D:

    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize(['nx', 'ny'], shapes)
    @pytest.mark.parametrize('fft_axes', fft_axeses, ids=[str(_) for _ in fft_axeses])
    def test_props(self, nx, ny, fft_axes: Tuple[int,...]|None, real_):

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        real = real_ == 'real'
        shape = (nx, ny)
      
        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealGrid2D(*shape, fft_axes=fft_axes)
                assert g.shape[last_fft_axis] == shape[last_fft_axis] + 1
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)
            else:
                g = RealGrid2D(*shape, fft_axes=fft_axes)

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
        assert g.precision == 'DOUBLE'
        assert g.fft_axes == fft_axes



    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize(['nx', 'ny'], shapes)
    @pytest.mark.parametrize('fft_axes', fft_axeses)
    def test_copy(self, nx, ny, fft_axes: Tuple[int,...]|None, real_):
        real = real_ == 'real'
        shape = (nx, ny) 

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)
            g = RealGrid2D(*shape, fft_axes=fft_axes)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real))

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.shape == h.shape
        assert g.rank == h.rank

        assert g.nx == h.nx
        assert g.ny == h.ny

        assert g.shape_k == h.shape_k
        assert g.fft_axes == h.fft_axes
        assert g.last_fft_axis == h.last_fft_axis
        assert g.numel == h.numel
        assert g.isreal == h.isreal
        assert g.precision == h.precision



    @pytest.mark.slow
    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize(['nx', 'ny'], shapes)
    @pytest.mark.parametrize('fft_axes', fft_axeses)
    def test_fft(self, nx, ny, fft_axes: Tuple[int,...]|None, real_):
        real = real_ == 'real'
        shape = (nx, ny) 

        if fft_axes is None:
            fft_axes = (0,1)
        last_fft_axis = fft_axes[-1]

        if not real:
            g = ComplexGrid2D(*shape, fft_axes=fft_axes)
        else:
            if shape[last_fft_axis] % 2 == 1:
                shape_ = list(shape)
                shape_[last_fft_axis] += 1
                shape = tuple(shape_)
            g = RealGrid2D(*shape, fft_axes=fft_axes)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = T.gen_array(shape, **ARRAY_GEN_CFG, complex=(not real))

            g.psi[...] = f
            assert np.array_equal(g.psi, f)

            g.fft() 
            fk = fft2(f, axes=fft_axes)
            if not real:
                assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore
            else:
                fk_r = rfft2(f, axes=fft_axes)
                assert T.adaptive_atol_allclose(g.psi_k, fk_r, **TOL) # type: ignore

            g.ifft()
            f = ifft2(fk, axes=fft_axes)
            assert T.adaptive_atol_allclose(g.psi, f, **TOL) # type: ignore

            if real:
                fr = irfft2(fk_r, axes=fft_axes) # type: ignore
                assert T.adaptive_atol_allclose(g.psi, fr, **TOL) # type: ignore


