from __future__ import annotations
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, TypeVar, overload
import numpy as np
import numpy.typing as npt

from ...core import PrecisionLike, get_real_dtype


T = TypeVar('T')


def gen_shapes(n: int, ranks: Sequence[int],
               max_numel: float, min_numel: float) -> List[Tuple[int,...]]:
    """Generate multiple shapes"""
    res = []
    for _ in range(n):
        rank = np.random.choice(ranks)
        shape = gen_random_sequence(rank, max_prod=max_numel, min_prod=min_numel, integ=True)
        res.append(shape)
    return res


def gen_fft_axes(rank: int):
    """Generate FFT axes"""
    n_ax = np.random.randint(1, rank+1)
    return tuple(np.random.choice(range(rank), replace=False, size=n_ax).tolist())


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


def gen_array(shape: Tuple[int,...], 
              min_offset: float = 0, max_offset: float = 0,
              min_scale: float = 1, max_scale: float = 1,
              complex: bool = False, precision: PrecisionLike = 'double'
              ):
    """
    Generate random array
    """
    if complex:
        a = gen_array(shape, min_offset, max_offset, min_scale, max_scale, precision=precision)
        b = gen_array(shape, min_offset, max_offset, min_scale, max_scale, precision=precision)
        return a + 1j*b

    dtype = get_real_dtype(precision)
    offset = np.random.rand() * (max_offset - min_offset) + min_scale
    logscale =  np.random.rand() * (np.log(max_scale) - np.log(min_scale)) + np.log(min_scale)
    return ((np.random.rand(*shape) + offset) * np.exp(logscale)).astype(dtype)


def random_permutation(seq: Sequence[T]) -> List[T]:
    indices = np.random.choice(range(len(seq)), size=len(seq), replace=False).tolist()
    return [seq[i] for i in indices]


