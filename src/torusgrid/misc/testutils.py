from typing import List, Sequence, Tuple
import numpy as np
import numpy.typing as npt



def prod(arr: Sequence[float]):
    """product of all the elements in an array"""
    r = 1
    for a in arr: r *= a
    return r


def gen_shape(rank: int, min_log_numel: float, max_log_numel: float):
    """Generate shape"""
    log_numel = np.random.rand() * (max_log_numel - min_log_numel) + min_log_numel

    seps = np.random.rand(rank-1).tolist()
    seps = [0] + sorted(seps) + [1]
    
    shape_ = []

    for i in range(rank):
        s = np.exp(log_numel * (seps[i+1] - seps[i]))
        shape_.append(int(s))

    return tuple(shape_)


def gen_shapes(n: int, ranks: Sequence[int],
               max_log_numel: float, min_log_numel: float) -> List[Tuple[int,...]]:
    """Generate multiple shapes"""
    res = []
    for _ in range(n):
        rank = np.random.choice(ranks)
        shape = gen_shape(rank, max_log_numel=max_log_numel, min_log_numel=min_log_numel)
        res.append(shape)
    return res


def gen_size(shape: Tuple[int,...], min, max):
    """
    Return a random size based on shape
    """
    rank = len(shape)
    size = []
    for _ in range(rank):
        size.append(np.random.rand()*(max-min) + min)
    return tuple(size)


def gen_random_sequence(length: int, min_prod: float, max_prod: float, *, integ: bool=False):
    """
    Genereate a sequence of fixed length such that min_prod < product of all elements < max_prod
    """
    min_log_prod = np.log(min_prod)
    max_log_prod = np.log(max_prod)

    log_prod = np.random.rand() * (max_log_prod - min_log_prod) + min_log_prod

    seps = np.random.rand(length-1).tolist()
    seps = [0] + sorted(seps) + [1]
    
    res = []
    for i in range(length):
        s = np.exp(log_prod * (seps[i+1] - seps[i]))
        if integ:
            s = round(s)
        res.append(s)
    return tuple(res)


def gen_fft_axes(rank: int):
    """Generate FFT axes"""
    n_ax = np.random.randint(1, rank+1)
    return tuple(np.random.choice(range(rank), replace=False, size=n_ax).tolist())


def float_tup_repr(tup, fmt: str='.3e'):
    return '(' + ','.join([f'{s:{fmt}}' for s in tup]) + ')'


def int_tup_repr(tup: Tuple[int,...]):
    return '(' + ','.join([str(s) for s in tup]) + ')'


def get_test_id(size, shape, fft_axes):
    """Test IDs"""
    return f'size={float_tup_repr(size)} shape={int_tup_repr(shape)} fft={int_tup_repr(fft_axes)}'


def gen_array(shape: Tuple[int,...], 
              min_offset: float = 0, max_offset: float = 0,
              min_scale: float = 1, max_scale: float = 1,
              complex: bool = False
              ):
    """
    Generate random array
    """
    if complex:
        a = gen_array(shape, min_offset, max_offset, min_scale, max_scale)
        b = gen_array(shape, min_offset, max_offset, min_scale, max_scale)
        return a + 1j*b

    offset = np.random.rand() * (max_offset - min_offset) + min_scale
    logscale =  np.random.rand() * (np.log(max_scale) - np.log(min_scale)) + np.log(min_scale)
    return (np.random.rand(*shape) + offset) * np.exp(logscale)


def adaptive_atol_allclose(f1: npt.ArrayLike, f2: npt.ArrayLike, rtol: float, *,
                           atol_factor: float = 1e-5):
    """
    Compare two arrays symmetrically
    """
    f1_ = np.asarray(f1)
    f2_ = np.asarray(f2)

    dev1 = np.sqrt(np.mean(np.abs(f1-np.mean(f1))**2))
    dev2 = np.sqrt(np.mean(np.abs(f1-np.mean(f1))**2))

    dev = np.sqrt((dev1**2+dev2**2)/2)

    atol = dev * atol_factor

    mag = np.sqrt((np.abs(f1_**2) + np.abs(f2_**2)) / 2)

    return np.all(np.abs(f1_ - f2_) <= rtol * mag + atol) # type: ignore


real_dtype_table = {
    'SINGLE': np.single,
    'DOUBLE': np.double,
    'LONGDOUBLE': np.longdouble
}


complex_dtype_table = {
    'SINGLE': np.csingle,
    'DOUBLE': np.cdouble,
    'LONGDOUBLE': np.clongdouble
}


nbytes_table = {'SINGLE': 4, 'DOUBLE': 8, 'LONGDOUBLE': 16}



