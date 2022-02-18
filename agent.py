import random
from abc import ABC, abstractmethod
from typing import Sequence, TypeVar

T = TypeVar('T')


class Agent(ABC):
    @abstractmethod
    def act(self, state, **kwargs):
        pass


class RandomAgent(Agent):
    def __init__(self, seed):
        random.seed(seed)

    def act(self, state, actions: Sequence[T]) -> T:
        return random.choice(actions)


# todo make torch model agent
class TorchModelAgent(Agent):
    pass
