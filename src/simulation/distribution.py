from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):

    @abstractmethod
    def draw(self) -> float:
        pass


class DeterministicDistribution(Distribution):

    def __init__(self, value: float) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Deterministic({self.value})"

    def draw(self) -> float:
        return self.value


class LogNormDistribution(Distribution):

    def __init__(self, shape: float, loc: float, scale: float) -> None:
        self.shape = shape
        self.loc = loc
        self.scale = scale

    def __repr__(self) -> str:
        return f"LogNorm(s={self.shape},loc={self.loc},scale={self.scale})"

    def draw(self) -> float:
        return (np.random.lognormal(0, self.shape, 1).item() * self.scale +
                self.loc)


_CLASS_MAPPING = {
    "deterministic": DeterministicDistribution,
    "lognorm": LogNormDistribution,
}


def create_distribution(type: str, *args, **kwargs) -> Distribution:
    return _CLASS_MAPPING[type](*args, **kwargs)
