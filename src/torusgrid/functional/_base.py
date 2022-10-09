from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy.typing as npt
import numpy as np
from ..misc.typing import generic
from ..fields import ComplexFieldND


T  = TypeVar('T', bound=ComplexFieldND)


@generic
class FreeEnergyFunctional(ABC, Generic[T]):
    '''
    Only makes sense when the field.psi can be interpreted as a density.
    '''
    @abstractmethod
    def derivative(self, field: T):
        raise NotImplementedError()

    @abstractmethod
    def free_energy_density(self, field: T) -> npt.NDArray:
        raise NotImplementedError()

    def free_energy(self, field: T) -> float:
        return np.sum(self.free_energy_density(field)) * field.dV

    def mean_free_energy_density(self, field: T) -> float:
        return np.mean(self.free_energy_density(field))

    def grand_potential_density(self, field: T, mu: float) -> npt.NDArray:
        return self.free_energy_density(field) - mu * field.psi

    def grand_potential(self, field: T, mu: float) -> float:
        return np.sum(self.grand_potential_density(field, mu)) * field.dV

    
    def mean_grand_potential_density(self, field: T, mu: float) -> float:
        return np.mean(self.free_energy_density(field))

