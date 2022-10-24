from __future__ import annotations
import pytest
from torusgrid.grids import ComplexGrid1D, RealGrid1D
from scipy.fft import fft, ifft, rfft, irfft
import numpy as np

isreal = ['real', 'complex']

ns = [
    1,2,
    3,4,
    5,8,13,16,
    17,18,23,25,32,
    39,48,49,50,59,64,
    75,79,82,94,126,128,
    150,199,200,241,250,256,
    304,311,320,396,418,447,494,512,
    552,559,590,621,640,718,844,932,1023,1024,
    2001,2048,3972,4096,4113,8192,12636,16384,32768,65535,65536
]


@pytest.mark.parametrize('real_', isreal)
class TestGrid1D:

    @pytest.mark.parametrize('n', ns)
    def test_props(self, n: int, real_: str):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n)
        else:
            if n % 2 == 1:
                with pytest.warns(UserWarning):
                    g = RealGrid1D(n)
                    assert g.n == n+1
                    n += 1
            else:
                g = RealGrid1D(n)

        assert g.n == n
        assert g.shape == (n,)
        assert g.rank == 1
        assert g.last_fft_axis == 0


    @pytest.mark.parametrize('n', ns)
    def test_fft(self, n: int, real_: bool):
        real = real_ == 'real'
        if not real:
            g = ComplexGrid1D(n)
        else:
            if n%2 == 1: n += 1
            g = RealGrid1D(n)

        N = 128

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=8)

        for _ in range(N):
            if not real:
                a = np.random.rand(n)
                b = np.random.rand(n)
                f = a + 1j*b
            else:
                f = np.random.rand(n)

            g.psi[...] = f
            assert np.all(g.psi == f)
            g.fft() 
            fk = fft(f)
            
            if real:
                fk_r = rfft(f)
                assert np.allclose(g.psi_k, fk_r, atol=1e-8, rtol=1e-8) # type: ignore
            else:
                assert np.allclose(g.psi_k, fk, atol=1e-8, rtol=1e-8) # type: ignore

            g.ifft()
            f = ifft(fk)

            if real:
                f_r = irfft(fk_r) # type: ignore
                assert np.allclose(g.psi, f_r, atol=1e-8, rtol=1e-8) # type: ignore

            assert np.allclose(g.psi, f, atol=1e-8, rtol=1e-8) # type: ignore



