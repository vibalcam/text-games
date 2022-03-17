import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, TypeVar, List

import torch

from models.models import load_model

T = TypeVar('T')


class Agent(ABC):
    @abstractmethod
    def act(self, state, **kwargs):
        pass


class RandomAgent(Agent):
    def __init__(self, seed: int = 123):
        random.seed(seed)

    def act(self, state:str, actions: Sequence[T]) -> T:
        return random.choice(actions)



class TorchRAgent(Agent):
    """
    Agent that uses a Pytorch model for decision taking.

    It chooses the action that provides a higher percentage of certainty of being a good decision.
    It tries to maximize: 
    
    `certainty = rand * random(0,1) + (1-rand) * predicted`
    """

    def __init__(self, model_path: Path, rand: float = 0, use_cpu: bool = False, seed: int = 123) -> None:
        """
        Initialize model
        :param model_path: path to the pytorch model to be used
        :param rand: importance of the random value. Should be a value between 0 and 1.
        :param use_cpu: whether to use the cpu for the model predictions
        :param seed: seed to use for the random generator
        """
        if not (0 <= rand <= 1):
            raise Exception("Rand must be a value between 0 and 1")

        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.model = load_model(model_path)[0].to(self.device)
        self.rand = rand
        self.generator = torch.random.manual_seed(seed)

    def _predict_label(self, state:str, actions: List[str]):
        """
        Predicts the labels from the state and actions given
        """
        # get percentages of certainty of action with good result
        # model run outputs B,1 -> pred B
        return self.model.run([state] * len(actions), actions, device=self.device,
                              return_percentages=True).cpu().detach()[:,0]

    def _decide(self, labels):
        """
        Decides which action to take from the set of labels per action
        """
        # add randomness to prediction
        pred = (self.rand) * torch.rand(labels.shape, generator=self.generator, dtype=torch.float) + (
                    1 - self.rand) * labels
        # choose option with highest certainty of good result
        return pred.argmax(0)

    def act(self, state:str, actions: List[str]) -> int:
        pred = self._predict_label(state=state, actions=actions)
        pred = self._decide(pred)
        return pred


class TorchFuncAgent(Agent):
    """
    Agent that uses a Pytorch model for decision taking.
    """

    def __init__(self, model_path: Path, rand: float = 0, use_cpu: bool = False, seed: int = 123) -> None:
        """
        Initialize model
        :param model_path: path to the pytorch model to be used
        :param rand: importance of the random value. Should be a value between 0 and 1.
        :param use_cpu: whether to use the cpu for the model predictions
        :param seed: seed to use for the random generator
        """
        if not (0 <= rand <= 1):
            raise Exception("Rand must be a value between 0 and 1")

        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.model = load_model(model_path)[0].to(self.device)
        self.rand = rand
        self.generator = torch.random.manual_seed(seed)

    def _predict_label(self, state:str, actions: List[str]):
        """
        Predicts the labels from the state and actions given
        """
        # get percentages of certainty of action with good result
        # model run outputs B,1 -> pred B
        return self.model.run([state] * len(actions), actions, device=self.device,
                              return_percentages=True).cpu().detach()[:,0]

    def _decide(self, labels):
        """
        Decides which action to take from the set of labels per action
        """
        # add randomness to prediction
        pred = (self.rand) * torch.rand(labels.shape, generator=self.generator, dtype=torch.float) + (
                    1 - self.rand) * labels
        # choose option with highest certainty of good result
        return pred.argmax(0)

    def act(self, state:str, actions: List[str]) -> int:
        pred = self._predict_label(state=state, actions=actions)
        pred = self._decide(pred)
        return pred