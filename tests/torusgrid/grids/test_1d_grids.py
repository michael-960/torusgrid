from __future__ import annotations
import pytest
from torusgrid.core.dtypes import FloatingPointPrecision, PrecisionLike, PrecisionStr
from torusgrid.grids import ComplexGrid1D, RealGrid1D
from scipy.fft import fft, ifft, rfft, irfft
import numpy as np

import torusgrid.misc.testutils as T

_cfg = T.get_config()
config = _cfg['grids']['1d']
gconfig = _cfg['global']

ARRAY_GEN_CFG = dict(
    min_offset=config['min_offset'], max_offset=config['max_offset'],
    min_scale=config['min_scale'], max_scale=config['max_scale']
)

def get_fft_tol(p: PrecisionLike):
    return dict(
            rtol1=T.floating_tol(p, cfg=gconfig['fft_tol']),
            rtol2=T.floating_tol(p, cfg=gconfig['fft_tol']))


shapes = T.gen_shapes(config['num_samples'], ranks=[1], 
                      min_numel=config['min_numel'],
                      max_numel=config['max_numel'])
ns = [shape[0] for shape in shapes]
reals = [['real','complex'][np.random.randint(0,2)] for _ in shapes]
precisions = [['single', 'double', 'longdouble'][np.random.randint(0,3)] for _ in shapes]

argvals = dict(n=ns, real_=reals, precision=precisions)


class TestGrid1D:
    """
    Test 1D complex and real grids functionalites
    """

    @T.parametrize('n', 'real_', 'precision', argvals=argvals)
    def test_props(self, n: int, real_: str, precision: PrecisionStr):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n, precision=precision)
            nk = n
        else:
            if n % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealGrid1D(n, precision=precision)
                    assert g.n == n+1
                    n += 1
            else:
                g = RealGrid1D(n, precision=precision)

            nk = n//2 + 1

        assert g.n == n
        assert g.shape == (n,)
        assert g.rank == 1
        assert g.last_fft_axis == 0
        assert g.fft_axes == (0,)
        assert g.precision is FloatingPointPrecision.cast(precision)
        assert g.psi.dtype == T.real_dtype(precision) if real else T.complex_dtype(precision)
        assert g.psi_k.dtype == T.complex_dtype(precision)
        assert g.shape_k == (nk,)


    @T.parametrize('n', 'real_', 'precision', argvals=argvals)
    def test_copy(self, n: int, real_: bool, precision: PrecisionStr):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n, precision=precision)
        else:
            if n%2 == 1: n += 1
            g = RealGrid1D(n, precision=precision)

        g.psi[...] = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real), precision=precision)

        h = g.copy()

        assert h.psi is not g.psi
        assert h.psi_k is not g.psi_k

        assert np.array_equal(h.psi, g.psi)
        assert np.array_equal(h.psi_k, g.psi_k)

        assert g.n == h.n

        assert T.same_meta(g, h)

    @pytest.mark.slow
    @T.parametrize('n', 'real_', 'precision', argvals=argvals)
    def test_fft(self, n: int, real_: bool, precision: PrecisionStr):
        real = real_ == 'real'

        if not real:
            g = ComplexGrid1D(n, precision=precision)
        else:
            if n%2 == 1: n += 1
            g = RealGrid1D(n, precision=precision)

        tol = get_fft_tol(g.precision)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=config['num_fft_threads'])

        for _ in range(config['num_fft_trials']):
            f = T.gen_array((n,), **ARRAY_GEN_CFG, complex=(not real), precision=precision)


            g.psi[...] = f
            assert np.array_equal(g.psi, f)
            g.fft() 
            fk = fft(f)

            if real:
                fk_r = rfft(f)
                assert T.adaptive_allclose(
                        g.psi_k, fk_r,  # type: ignore
                        **tol)
            else:
                assert T.adaptive_allclose(
                        g.psi_k, fk,  # type: ignore
                        **tol)

            g.ifft()
            f = ifft(fk)

            if real:
                f_r = irfft(fk_r) # type: ignore
                assert T.adaptive_allclose(
                        g.psi, f_r, # type: ignore
                        **tol) 
            assert T.adaptive_allclose(
                    g.psi, f, # type: ignore 
                    **tol)

            """
            Finally, to make sure that T.adaptive_allclose is working as
            intended
            """
            assert not T.adaptive_allclose(
                    g.psi, 
                    T.gen_array(g.shape, **ARRAY_GEN_CFG),
                    **tol)





