from __future__ import annotations
from typing import Any, Dict, Union
from decimal import Decimal
import dataclasses

from kubernetes.utils import parse_quantity


@dataclasses.dataclass
class Resources:
    cpu: Decimal = Decimal(0)
    memory: Decimal = Decimal(0)

    def __iadd__(self, other):
        self.cpu += other.cpu
        self.memory += other.memory
        return self

    def __isub__(self, other):
        self.cpu -= other.cpu
        self.memory -= other.memory
        return self

    def __add__(self, other: Resources):
        return Resources(cpu=(self.cpu + other.cpu),
                         memory=(self.memory + other.memory))

    def __sub__(self, other: Resources):
        return Resources(cpu=(self.cpu - other.cpu),
                         memory=(self.memory - other.memory))

    def __repr__(self):
        return f"Resources(cpu={self.cpu}, memory={self.memory})"

    def __lt__(self, other: Resources):
        return self.cpu < other.cpu and self.memory < other.memory

    def __le__(self, other: Resources):
        return self.cpu <= other.cpu and self.memory <= other.memory

    def __gt__(self, other: Resources):
        return self.cpu > other.cpu and self.memory > other.memory

    def __ge__(self, other: Resources):
        return self.cpu >= other.cpu and self.memory >= other.memory

    def __truediv__(self, other: Resources):
        return Resources(cpu=(self.cpu / other.cpu),
                         memory=(self.memory / other.memory))

    def __floordiv__(self, other: Resources):
        return Resources(cpu=(self.cpu // other.cpu),
                         memory=(self.memory // other.memory))

    def __neg__(self):
        return Resources(cpu=-self.cpu, memory=-self.memory)


def parse_resource_dict(resource_dict: Dict[str, Union[int, str]]) -> Resources:
    return Resources(cpu=parse_quantity(resource_dict["cpu"]),
                     memory=parse_quantity(resource_dict["memory"]))


def parse_resources(resources: Dict[str, Any]) -> Resources:
    return Resources(cpu=parse_quantity(resources["cpu"]),
                     memory=parse_quantity(resources["memory"]))
