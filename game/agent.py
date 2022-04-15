import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, TypeVar, List

import torch
from overrides import EnforceOverrides, overrides

from models.models import load_model
from game.simulator import GraphSimulator

T = TypeVar('T')


class Agent(ABC):
    @abstractmethod
    def act(self, state:str, **kwargs):
        pass


class RandomAgent(Agent):
    def __init__(self, seed: int = None):
        random.seed(seed)

    @overrides(check_signature=False)
    def act(self, state:str, actions: Sequence[T], **kwargs) -> T:
        return random.choice(actions)


class LabelPredictor(ABC):
    @abstractmethod
    def predict_label(self, state:str, actions: List[str]) -> torch.Tensor:
        """
        Predicts the labels from the state and actions given
        """
        pass


class DecisionMaker(ABC):
    @abstractmethod
    def decide(self, pred: torch.Tensor, **kwargs) -> int:
        pass


class LabelDecisorAgent (Agent):
    def __init__(self, label_predictor: LabelPredictor, decisor: DecisionMaker):
        self.label_predictor = label_predictor
        self.decisor = decisor

    @overrides(check_signature=False)
    def act(self, state:str, actions: List[str], **kwargs):
        return self.decisor.decide(self.label_predictor.predict_label(state=state, actions=actions), **kwargs)


class GraphLabelLoader(LabelPredictor):
    def __init__(self, simulator: GraphSimulator) -> None:
        super().__init__()
        self.simulator = simulator

    @overrides
    def predict_label(self, **kwargs) -> torch.Tensor:
        extras = self.simulator.read()[2]
        return torch.as_tensor([extras[GraphSimulator.ATTR_PRED]],dtype=torch.float)


class TorchLabelPredictor(LabelPredictor):
    def __init__(self, model_path: Path, use_cpu: bool = False) -> None:
        """
        Uses a Pytorch model for predicting the labels

        :param model_path: path to the pytorch model to be used
        :param use_cpu: whether to use the cpu for the model predictions
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.model = load_model(model_path)[0].eval().to(self.device)

    @overrides
    def predict_label(self, state:str, actions: List[str]) -> torch.Tensor:
        # get percentages of certainty of action
        # model run outputs B,1 -> pred B
        return self.model.run([state] * len(actions), actions, return_percentages=True).cpu().detach()[:,0]


class RDecisionMaker(DecisionMaker):
    def __init__(self, rand: float = 0, seed: int = 4444) -> None:
        """
        It chooses the action that provides a higher percentage of certainty of being a good decision.
        It tries to maximize:

        `certainty = rand * random(0,1) + (1-rand) * predicted`

        :param rand: importance of the random value. Should be a value between 0 and 1.
        :param seed: seed to use for the random generator
        """
        if not (0 <= rand <= 1):
            raise Exception("Rand must be a value between 0 and 1")
        self.rand = rand
        self.generator = torch.Generator().manual_seed(seed)

    @overrides
    def decide(self, pred:torch.Tensor, **kwargs) -> int:
        # add randomness to prediction
        pred = self.rand * torch.rand(pred.shape, generator=self.generator, dtype=torch.float) + (1 - self.rand) * pred
        # choose option with highest certainty of good result
        return pred.argmax().item()


# class TorchFuncAgent(TorchAgent):
#     """
#     Agent that uses a Pytorch model for decision taking.
#     """
#
#     def __init__(self,
#         model_path: Path,
#         alpha: torch.Tensor,
#         beta: torch.Tensor,
#         use_cpu: bool = False,
#         seed: int = 123
#     ) -> None:
#         super().__init__(model_path=model_path, use_cpu=use_cpu, seed=seed)
#
#         self.alpha = alpha
#         self.beta = beta
#         self.past_dec = None
#
#     def _func_labels(self, labels):
#         pass
#
#     def _decide(self, labels) -> int:
#         if self.past_dec is None:
#             p = self._func_labels(labels)
#         else:
#             p = (1 - self.beta - self.alpha) * self._func_labels(labels) + \
#                 self.alpha * self.past_dec + \
#                 self.beta * (1 - self.past_dec)
#
#         """
#          todo: no es un unico label, sino por accion
#          solucion posible
#          - calcular p para cada accion
#          - normalizar sobre 1 y obtener probabilidades acumuladas por opcion
#          - uniforme de 0 a 1
#          - elegir aquella eleccion donde la uniforme haya caido (tramo donde haya caido)
#         """
#
#         self.past_dec = p.argmax(0)
#         return self.past_dec


if __name__ == '__main__':
    from game.simulator import load_simulator_yarn, GraphSimulator

    # simulator
    simulator = load_simulator_yarn()

    # agent
    # agent = RandomAgent(4444)
    agent = LabelDecisorAgent(
        label_predictor=TorchLabelPredictor(
            model_path=Path('./models/tmp/saved_good/adamw_max_val_acc_8_False_125,[20],[20]_0.001'),
            use_cpu=True,
        ),
        decisor=RDecisionMaker(
            rand=0,
            seed=4444,
        )
    )

    # run game
    state, choices, reward = simulator.read()
    while not simulator.is_finished():
        choice = agent.act(state, choices)
        print(f"{state}\n{choice}\n\n")
        state, choices, extras = simulator.transition(choice)
        # print((state, choices, extras))
