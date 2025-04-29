from abc import ABC, abstractmethod
from . import lazy
from .fuse import *

class AbstractTensor(ABC):
    @abstractmethod
    def shape(self):
        """Return the shape of the tensor."""
        pass

    @abstractmethod
    def dtype(self):
        """Return the data type of the tensor."""
        pass

    @abstractmethod
    def to_numpy(self):
        """Convert the tensor to a NumPy array."""
        pass

    @abstractmethod
    def __add__(self, other):
        """Define addition for tensors."""
        pass

    @abstractmethod
    def __mul__(self, other):
        """Define multiplication for tensors."""
        pass

def prod(arr: Tensor, dims):
    return compute(lazy.prod(lazy.lazy(arr), dims))
