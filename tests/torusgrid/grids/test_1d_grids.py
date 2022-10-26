from __future__ import annotations
import pytest
from torusgrid.grids import ComplexGrid1D, RealGrid1D
from scipy.fft import fft, ifft, rfft, irfft
import numpy as np

import torusgrid.misc.testutils as T


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

shapes = T.gen_shapes(NUM_SHAPES, ranks=[1], 
                      min_log_numel=np.log(MIN_NUMEL),
                      max_log_numel=np.log(MAX_NUMEL))

ns = [shape[0] for shape in shapes]


class TestGrid1D:
    """
    Test 1D complex and real grids functionalites
    """

    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize('n', ns)
    def test_props(self, n: int, real_: str):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n)
            nk = n
        else:
            if n % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealGrid1D(n)
                    assert g.n == n+1
                    n += 1
            else:
                g = RealGrid1D(n)

            nk = n//2 + 1

        assert g.n == n
        assert g.shape == (n,)
        assert g.rank == 1
        assert g.last_fft_axis == 0
        assert g.fft_axes == (0,)
        assert g.precision == 'DOUBLE'
        assert g.shape_k == (nk,)


    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize('n', ns)
    def test_copy(self, n: int, real_: bool):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n)
        else:
            if n%2 == 1: n += 1
            g = RealGrid1D(n)

        g.psi[...] = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real))

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.shape == h.shape
        assert g.rank == h.rank
        assert g.n == h.n

        assert g.shape_k == h.shape_k
        assert g.last_fft_axis == h.last_fft_axis
        assert g.fft_axes == h.fft_axes
        assert g.numel == h.numel
        assert g.isreal == h.isreal
        assert g.precision == h.precision


    @pytest.mark.slow
    @pytest.mark.parametrize('real_', isreal)
    @pytest.mark.parametrize('n', ns)
    def test_fft(self, n: int, real_: bool):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n)
        else:
            if n%2 == 1: n += 1
            g = RealGrid1D(n)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real))

            g.psi[...] = f
            assert np.array_equal(g.psi, f)
            g.fft() 
            fk = fft(f)
            
            if real:
                fk_r = rfft(f)
                assert T.adaptive_atol_allclose(g.psi_k, fk_r, **TOL) # type: ignore
            else:
                assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore

            g.ifft()
            f = ifft(fk)

            if real:
                f_r = irfft(fk_r) # type: ignore
                assert T.adaptive_atol_allclose(g.psi, f_r, **TOL) # type: ignore

            assert T.adaptive_atol_allclose(g.psi, f, **TOL) # type: ignore


            """
            Finally, to make sure that T.adaptive_atol_allclose is working as
            intended
            """
            assert not T.adaptive_atol_allclose(
                    g.psi, 
                    T.gen_array(g.shape, MIN_OFFSET, MAX_OFFSET, MIN_SCALE, MAX_SCALE),
                    **TOL)


