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


class TorchAgent(Agent):
    """
    Agent that uses a Pytorch model for decision taking.
    """

    def __init__(self, model_path: Path, use_cpu: bool = False, seed: int = 123) -> None:
        """
        Initialize model
        :param model_path: path to the pytorch model to be used
        :param use_cpu: whether to use the cpu for the model predictions
        :param seed: seed to use for the random generator
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.model = load_model(model_path)[0].to(self.device)
        self.generator = torch.Generator().random.manual_seed(seed)

    def _predict_label(self, state:str, actions: List[str]):
        """
        Predicts the labels from the state and actions given
        """
        # get percentages of certainty of action with good result
        # model run outputs B,1 -> pred B
        return self.model.run([state] * len(actions), actions, device=self.device,
                              return_percentages=True).cpu().detach()[:,0]

    @abstractmethod
    def _decide(self, labels, **kwargs) -> int:
        """
        Decides which action to take from the set of labels per action
        """
        pass

    def act(self, state:str, actions: List[str], **kwargs) -> int:
        pred = self._predict_label(state=state, actions=actions)
        pred = self._decide(pred, **kwargs)
        return pred


class TorchRAgent(TorchAgent):
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

        super().__init__(model_path=model_path, use_cpu=use_cpu, seed=seed)
        self.rand = rand

    def _decide(self, labels, **kwargs) -> int:
        # add randomness to prediction
        # todo probar si funciona
        pred = (self.rand) * torch.rand(labels.shape, generator=self.generator, dtype=torch.float) + (
                    1 - self.rand) * labels
        # choose option with highest certainty of good result
        return pred.argmax(0)


class TorchFuncAgent(TorchAgent):
    """
    Agent that uses a Pytorch model for decision taking.
    """

    def __init__(self, 
        model_path: Path, 
        alpha: torch.Tensor, 
        beta: torch.Tensor,
        use_cpu: bool = False, 
        seed: int = 123
    ) -> None:
        super().__init__(model_path=model_path, use_cpu=use_cpu, seed=seed)

        self.alpha = alpha
        self.beta = beta
        self.past_dec = None

    def _func_labels(self, labels):
        pass

    def _decide(self, labels) -> int:
        if self.past_dec is None:
            p = self._func_labels(labels)
        else:
            p = (1 - self.beta - self.alpha) * self._func_labels(labels) + \
                self.alpha * self.past_dec + \
                self.beta * (1 - self.past_dec)

        """
         todo: no es un unico label, sino por accion
         solucion posible
         - calcular p para cada accion
         - normalizar sobre 1 y obtener probabilidades acumuladas por opcion
         - uniforme de 0 a 1
         - elegir aquella eleccion donde la uniforme haya caido (tramo donde haya caido)
        """
        
        self.past_dec = p.argmax(0)
        return self.past_dec


if __name__ == '__main__':
    from simulator import YarnSimulator

    simulator = YarnSimulator(text_unk_macro="")

    # create agent
    agent = RandomAgent(15)
    # agent = TorchRAgent(
    #     model_path=Path('models/saved/adamw_max_val_acc_8_False_125,[20],[20]_0.001'),
    #     rand=0,
    #     use_cpu=False,
    #     seed=123,
    # )

    # run game
    state, choices, reward = simulator.read()
    while not simulator.is_finished():
        choice = agent.act(state, choices)
        print(f"{state}\n{choice}\n\n")
        state, choices, reward = simulator.transition(choice)
        # print((state, choices, reward))
