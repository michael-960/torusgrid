from __future__ import annotations
from typing import Tuple
import pytest
from torusgrid.grids import ComplexGrid2D , RealGrid2D
from scipy.fft import fft2, ifft2, rfft2, irfft2
import numpy as np

isreal = ['real', 'complex']

shapes = [
    (2,3), (2,4), (3,4),
    (4,8), (8,2), (7,8), (8,8),
    (5,9), (6,11), (13,1), (13,5), (16,16),
    (32,16), (16,32) ,(32,32),
    (64,16), (64,7), (23,64), (23,63), (64,64),
    (96,16), (32,96), (80, 88), (79, 101), (128,64), (128,128),
    (256,72), (32,255), (16,256), (32,256), (64,256), (256,128), (256,256),
    (300,300), (312,312), (256,512), (32,512), (64,512), (96,512), (80,512), (512,80), (512,512),
    (41,515), (612, 9), (667,668), (901,16), (1024,16), (1023, 16), (16,1024), (32, 1024), (512,1024),
    (2048,32), (2048,4), (4,2048), (2048,128), (2048,512), (2048, 47), (3,2047),
    (4096,8), (6144, 4), (3613,21),
    (8192,3), (8192,8),
    (16384,4),
    ]

fft_axeses = [None, (0,1), (0,), (1,), (1,0)]

@pytest.mark.parametrize('real_', isreal)
class TestGrid2D:
    @pytest.mark.parametrize(['nx', 'ny'], shapes)
    @pytest.mark.parametrize('fft_axes', fft_axeses)
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

        g.initialize_fft()

        assert g._fft is not None
        assert g._ifft is not None

        assert g.last_fft_axis == fft_axes[-1]
           
        assert g.nx == shape[0]
        assert g.ny == shape[1]

        if not real:
            shape_k = list(shape)
        else:
            shape_k = list(shape)
            shape_k[last_fft_axis] //= 2
            shape_k[last_fft_axis] += 1

        assert g.psi_k.shape == tuple(shape_k)


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

        N = 128

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=4)

        for _ in range(N):
            if not real:
                a = np.random.rand(*shape) 
                b = np.random.rand(*shape)
                f = a + 1j*b
            else:
                f = np.random.rand(*shape)

            g.psi[...] = f
            assert np.all(g.psi == f)

            g.fft() 
            fk = fft2(f, axes=fft_axes)
            if not real:
                assert np.allclose(g.psi_k, fk, atol=1e-8, rtol=1e-8) # type: ignore
            else:
                fk_r = rfft2(f, axes=fft_axes)
                assert np.allclose(g.psi_k, fk_r, atol=1e-8, rtol=1e-8) # type: ignore

            g.ifft()
            f = ifft2(fk, axes=fft_axes)
            assert np.allclose(g.psi, f, atol=1e-8, rtol=1e-8) # type: ignore

            if real:
                fr = irfft2(fk_r, axes=fft_axes) # type: ignore
                assert np.allclose(g.psi, fr, atol=1e-8, rtol=1e-8) # type: ignore



