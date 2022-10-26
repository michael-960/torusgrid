from torusgrid.fields import RealField
from torusgrid.misc import testutils as T
import numpy as np
import pytest
from scipy.fft import rfftn, irfftn, fftn, ifftn


NUM_SHAPES = 200
NUM_FFT_OPERATIONS = 128
NUM_FFT_THREADS = 8

NUM_FFTSPEED_OPERATIONS = 128

NUM_SETSIZE_TRIALS = 8

FFT_RTOL = 1e-10
FFT_ATOL_FACTOR = 1e-8

MAX_OFFSET = 0
MIN_OFFSET = -1

MAX_SCALE = 1e9
MIN_SCALE = 1e-9

MIN_NUMEL = 1
MAX_NUMEL = 1e5

MIN_VOL = 1e-10
MAX_VOL = 1e10


TOL = dict(rtol=FFT_RTOL, atol_factor=FFT_ATOL_FACTOR)

ARRAY_GEN_CFG = dict(
    min_offset=MIN_OFFSET, max_offset=MAX_OFFSET,
    min_scale=MIN_SCALE, max_scale=MAX_SCALE
)

shapes = T.gen_shapes(NUM_SHAPES, ranks=[1]*2 + [2]*5 + [3]*3 + [4,5,6,7,8], 
                      min_log_numel=np.log(MIN_NUMEL), max_log_numel=np.log(MAX_NUMEL))

sizes = [T.gen_random_sequence(len(shape), min_prod=MIN_VOL, max_prod=MAX_VOL) for shape in shapes]

fft_axeses = [T.gen_fft_axes(len(shape)) for shape in shapes]
_test_ids = [T.get_test_id(size,shape,fft_ax) for size,shape,fft_ax in zip(sizes, shapes, fft_axeses)]


def parametrize(f):
   return pytest.mark.parametrize(['size', 'shape', 'fft_axes'], zip(sizes, shapes, fft_axeses), ids=_test_ids)(f)


class TestRealFields:

    @parametrize
    def test_props(self, size, shape, fft_axes): 
        
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            with pytest.warns(UserWarning):
                g = RealField(size=size, shape=shape, fft_axes=fft_axes)

            assert g.shape[g.last_fft_axis] == shape[last_fft_axis] + 1
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)
        else:
            g = RealField(size=size, shape=shape, fft_axes=fft_axes)

        shape_k_ = list(shape)
        shape_k_[last_fft_axis] //= 2
        shape_k_[last_fft_axis] += 1
        shape_k = tuple(shape_k_)

        
        with pytest.warns(np.ComplexWarning):
            g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG, complex=True)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)

        assert g.shape == shape
        assert g.psi.shape == shape
        assert g.psi_k.shape == shape_k
        assert g.rank == len(shape)

        assert np.array_equal(g.size, size)

        assert g.numel == T.prod(shape)
        assert g.r.shape[0] == len(shape)
        assert g.r.shape[1:] == shape

        assert g.fft_axes == fft_axes
        assert g.last_fft_axis == fft_axes[-1]

        assert g.shape_k == g.k.shape[1:]

        assert np.size(g.r) == g.rank * T.prod(shape)
        assert g.volume == T.prod(g.size.tolist())
        
        
        r = []
        for l, n in zip(size, shape):
            r.append(np.linspace(0, l, n+1)[:-1])

        r = np.stack(np.meshgrid(*r, indexing='ij'), axis=0)
        assert T.adaptive_atol_allclose(r, g.r, **TOL)

        k = []
        for axis, (L, N) in enumerate(zip(size, shape)):
            dk = np.pi*2 / L
            M = N // 2
            _k = [i*dk for i in range(M)] + [i*dk for i in range(-(N-M), 0)]

            if axis == last_fft_axis:
                assert N % 2 == 0
                _k = [i*dk for i in range(M+1)]

            k.append(np.array(_k))

        k = np.stack(np.meshgrid(*k, indexing='ij'), axis=0)

        assert T.adaptive_atol_allclose(k, g.k, **TOL)
        k2 = np.sum(k**2, axis=0)

        assert T.adaptive_atol_allclose(k2, g.k2, **TOL)
        assert T.adaptive_atol_allclose(k2**2, g.k4, **TOL)
        assert T.adaptive_atol_allclose(k2**3, g.k6, **TOL)

        assert not T.adaptive_atol_allclose(k2**3.001, g.k6, **TOL)

        assert g._fft is None
        assert g._ifft is None
        assert g.isreal


    @parametrize
    def test_copy(self, size, shape, fft_axes):
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)
        g = RealField(size=size, shape=shape, fft_axes=fft_axes)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)

        h = g.copy()

        assert np.array_equal(g.psi, h.psi)
        assert g.psi is not h.psi
        assert g.psi_k is not h.psi_k

        assert g.rank == h.rank
        assert g.shape == h.shape
        assert g.shape_k == h.shape_k
        assert g.fft_axes == h.fft_axes
        assert g.isreal == h.isreal

        assert g.precision == h.precision

        assert g.numel == h.numel
        assert g.volume == h.volume

        assert np.array_equal(g.size, h.size)

        assert np.array_equal(g.r, h.r)
        assert np.array_equal(g.k, h.k)
        assert np.array_equal(g.k2, h.k2)
        assert np.array_equal(g.k4, h.k4)
        assert np.array_equal(g.k6, h.k6)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)

        assert not np.array_equal(g.psi, h.psi)


    @parametrize 
    def test_set_size(self, size, shape, fft_axes):
        last_fft_axis = fft_axes[-1]
        if shape[last_fft_axis] % 2 == 1:
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)
        g = RealField(size=size, shape=shape, fft_axes=fft_axes)

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)
        
        for _ in range(NUM_SETSIZE_TRIALS):
            size_ = T.gen_random_sequence(len(size), min_prod=MIN_VOL, max_prod=MAX_VOL)
            g.set_size(size_)

            assert T.adaptive_atol_allclose(g.size, size_, **TOL) # type: ignore

            r = []
            for l, n in zip(size_, shape):
                r.append(np.linspace(0, l, n+1)[:-1])

            r = np.stack(np.meshgrid(*r, indexing='ij'), axis=0)
            assert T.adaptive_atol_allclose(r, g.r, **TOL)

            k = []
            for axis, (L, N) in enumerate(zip(size_, shape)):
                dk = np.pi*2 / L
                M = N // 2
                _k = [i*dk for i in range(M)] + [i*dk for i in range(-(N-M), 0)]

                if axis == last_fft_axis:
                    assert N % 2 == 0
                    _k = [i*dk for i in range(M+1)]

                k.append(np.array(_k))

            k = np.stack(np.meshgrid(*k, indexing='ij'), axis=0)

            assert T.adaptive_atol_allclose(k, g.k, **TOL)
            k2 = np.sum(k**2, axis=0)

            assert T.adaptive_atol_allclose(k2, g.k2, **TOL)
            assert T.adaptive_atol_allclose(k2**2, g.k4, **TOL)
            assert T.adaptive_atol_allclose(k2**3, g.k6, **TOL)


    @parametrize
    @pytest.mark.parametrize('precision', ['SINGLE', 'DOUBLE', 'LONGDOUBLE', 'single', 'double', 'longdouble'])
    # @pytest.mark.parametrize('precision', ['single', 'DOUBLE', 'LONGDOUBLE', 'single', 'double', 'longdouble'])
    def test_precision(self, size, shape, fft_axes, precision):
        """
        Test whether the field has attributes with the appropriate precision
        """
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)


        if precision.upper() == 'SINGLE':
            l = min(size)
            n = max(shape)
            dk = np.pi*2 / l
            k = dk * n / 2

            if k**6 > 1e38 or k**6 < 1e-49:
                """
                overflow error
                """
                pytest.skip()


        g = RealField(size=size, shape=shape, 
                      fft_axes=fft_axes,
                      precision=precision)

        precision = precision.upper()
      
        assert g.psi.dtype == T.real_dtype_table[precision]
        assert g.psi.dtype != T.complex_dtype_table[precision]
      
        assert g.psi_k.dtype == T.complex_dtype_table[precision]
      
        assert g.psi.nbytes == g.numel * T.nbytes_table[precision] * 1
       
        assert g.psi_k.nbytes == T.prod(g.shape_k) * T.nbytes_table[precision] * 2
       
        assert g.size.dtype == T.real_dtype_table[precision]
        assert g.r.dtype == T.real_dtype_table[precision]
        assert g.k.dtype == T.real_dtype_table[precision]

        assert g.k2.dtype == T.real_dtype_table[precision]
        assert g.k4.dtype == T.real_dtype_table[precision]
        assert g.k6.dtype == T.real_dtype_table[precision]

        assert g.precision == precision

        g.psi[...] = T.gen_array(shape, **ARRAY_GEN_CFG)
        assert g.psi.dtype == T.real_dtype_table[precision]


    @pytest.mark.slow 
    @parametrize
    def test_fft(self, size, shape, fft_axes):
        """Test FFT"""
        last_fft_axis = fft_axes[-1]

        if shape[last_fft_axis] % 2 == 1:
            shape_ = list(shape)
            shape_[last_fft_axis] += 1
            shape = tuple(shape_)
        g = RealField(size=size, shape=shape, fft_axes=fft_axes)

        with pytest.raises(TypeError):
            g.fft()

        g.initialize_fft(threads=NUM_FFT_THREADS)

        for _ in range(NUM_FFT_OPERATIONS):
            f = np.random.rand(*shape)

            g.psi[...] = f

            assert np.array_equal(g.psi, f)

            g.fft()
            fk = rfftn(f, axes=fft_axes)
            fk_ = fftn(f, axes=fft_axes)
            assert T.adaptive_atol_allclose(g.psi_k, fk, **TOL) # type: ignore

            f_ = irfftn(fk, axes=fft_axes)
            f__ = ifftn(fk_, axes=fft_axes)

            assert T.adaptive_atol_allclose(g.psi, f_, **TOL) # type: ignore
            assert T.adaptive_atol_allclose(g.psi, f__, **TOL) # type: ignore


