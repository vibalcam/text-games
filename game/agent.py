import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, TypeVar, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from models.models import load_model
from overrides import EnforceOverrides, overrides
import pydtmc
from numpy.random import default_rng

from game.simulator import GraphSimulator

T = TypeVar('T')


######## AGENTS INTERFACE ########

class Agent(ABC):
    """
    Interface for agent (player)
    """
    
    @abstractmethod
    def act(self, state:str, **kwargs) -> Union[int,str]:
        """
        Takes a certain action

        :param str state: text description for the state (context)
        """
        pass


class LabelPredictor(ABC):
    @abstractmethod
    def predict_label(self, state:str, actions: List[str]) -> torch.Tensor:
        """
        Predicts the labels from the state and actions given

        :param str state: text description for the current state/context
        :param List[str] actions: list of actions that can be taken
        :return torch.Tensor: tensor (actions, labels) with the predicted probabilities of the labels for each action used for decision making
        """
        pass


class DecisionMaker(ABC):
    @abstractmethod
    def decide(self, pred: torch.Tensor, **kwargs) -> int:
        """
        Decides which action to take given a set of predicted labels

        :param torch.Tensor pred: tensor (actions, labels) with the predicted probabilities of the labels
        :return int: the action to take
        """
        pass


######## AGENTS ########

class RandomAgent(Agent):
    def __init__(self, seed: int = None):
        """
        An agent that takes decisions randomly

        :param int seed: seed of the random agent, defaults to None
        """
        random.seed(seed)

    @overrides(check_signature=False)
    def act(self, state:str, actions: Sequence[T], **kwargs) -> T:
        return random.choice(actions)


class MarkovChainAgent(Agent):
    def __init__(self, transitions: np.array, agents: List[Agent], initial_agent:int = 0, seed:int = None):
        """
        Markov Chain of agents

        :param np.array transitions: matrix of transition probabilities. 
            Element (i,j) is the probability that, being in state i, the agent will transition to state j
        :param List[Agent] agents: list of agents. Each agent is a state of the Markov Chain of agents
        :param int initial_agent: initial agent/state, defaults to 0
        :param int seed: seed for reproducibility
        """
        super().__init__()
        self.mc = pydtmc.MarkovChain(p=transitions, states=agents)
        self.current = agents[initial_agent]
        self.seed = seed

    @overrides(check_signature=False)
    def act(self, state:str, **kwargs):
        res = self.current.act(state, **kwargs)
        self.current = self.mc.next_state(initial_state=self.current, seed=self.seed)
        return res


class LabelDecisorAgent(Agent):
    def __init__(self, label_predictor: LabelPredictor, decision_maker: DecisionMaker):
        """
        Agent that takes decision by extracting a set of labels and taking decisions with them

        :param LabelPredictor label_predictor: label predictor that extracts the labels
        :param DecisionMaker decision_maker: decision_maker that takes decisions from a set of labels
        """
        self.label_predictor = label_predictor
        self.decision_maker = decision_maker

    @overrides(check_signature=False)
    def act(self, state:str, actions: List[str], **kwargs):
        return self.decision_maker.decide(self.label_predictor.predict_label(state=state, actions=actions), **kwargs)


######## LABEL PREDICTORS ########

class GraphLabelLoader(LabelPredictor):
    def __init__(self, simulator: GraphSimulator) -> None:
        """
        Extracts the predicted set of labels from the graph in the simulator

        :param GraphSimulator simulator: simulator of the game environment
        """
        super().__init__()
        self.simulator = simulator

    @overrides(check_signature=False)
    def predict_label(self, **kwargs) -> torch.Tensor:
        """
        Obtains the predicted labels from the graph simulator

        :return torch.Tensor: predicted labels (actions, labels)
        """
        preds = [self.simulator.actions[k][GraphSimulator.ATTR_EXTRAS][GraphSimulator.ATTR_PRED] for k in self.simulator.showed_actions]
        return torch.as_tensor(preds, dtype=torch.float)


class TorchLabelPredictor(LabelPredictor):
    def __init__(self, model_path: Path, use_cpu: bool = False) -> None:
        """
        Uses a Pytorch model for predicting the labels

        :param model_path: path to the pytorch model to be used
        :param use_cpu: whether to use the cpu for the model predictions
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.model = load_model(model_path)[0].eval().to(self.device)
        self._last_probs = torch.as_tensor(0, dtype=torch.float)

    @property
    def last_probs(self) -> torch.Tensor:
        return self._last_probs

    @overrides(check_signature=False)
    def predict_label(self, state:str, actions: List[str]) -> torch.Tensor:
        # get percentages of certainty of action
        # model run outputs B,labels -> pred B,labels
        self._last_probs = self.model.run([state] * len(actions), actions, return_percentages=True).cpu().detach()
        return self.last_probs


######## DECISION MAKERS ########

class RDecisionMaker(DecisionMaker):
    def __init__(self, rand: float = 0, seed: Optional[int] = 4444) -> None:
        """
        WARNING: ONLY USED FOR GOOD/BAD DECISIONS WITH LABELS DIMENSION OF (actions, 1)

        It chooses the action that provides a higher percentage of certainty of being a good decision.
        It tries to maximize:

        `certainty = rand * random(0,1) + (1-rand) * predicted`

        :param rand: importance of the random value. Should be a value between 0 and 1.
        :param seed: seed to use for the random generator
        """
        if not (0 <= rand <= 1):
            raise Exception("Rand must be a value between 0 and 1")
        self.rand = rand
        self.generator = torch.Generator()
        if seed is None:
            # random seed
            self.generator.seed()
        else:
            # given seed
            self.generator = self.generator.manual_seed(seed)

    @overrides(check_signature=False)
    def decide(self, pred:torch.Tensor, **kwargs) -> int:
        # add randomness to prediction
        pred = self.rand * torch.rand(pred.shape, generator=self.generator, dtype=torch.float) + (1 - self.rand) * pred
        # choose option with highest certainty of good result
        return pred.argmax().item()


class BehavioralDecisionMaker(DecisionMaker):
    def __init__(self, weight_funcs: List[Callable[[torch.Tensor, torch.Tensor], float]], memory_steps:int = 0, seed:int = None, deterministic:bool = False):
        """
        Decision maker with a certain behavior profile and memory-aware.

        Calculates the probability of taking a certain action according to the alignment between the personality traits/labels of the action and
        its behavior profile. This probability is computed by weighting the predicted labels. These weights are obtained 
        taking into account past decisions and using a given function that characterizes the behavior profile.

        Note that, if weight w for label r is w>0, it encourages this behavior; w=0, does not care; w<0, discourages this behavior

        :param List[Callable[[List[torch.Tensor, torch.Tensor]], float]] weight_funcs: list of functions to obtain the weights for each label. 
            The number of functions must equal the number of labels. The weight for label `r` given function `f_r`, will be obtained
            by computing `f(M,S)` with `M` being the memory matrix and `S` being the steps matrix. This function must return a float value.
            `M` is a matrix (memory_steps x len(weight_funcs)) where the first row is 1 and the rest of rows `i` are the labels of the 
            action taken `i-1` steps ago.
            `S` is a column vector (memory_steps, 1) where row `i` is 1 if an action has been taken for step `i-1`, and 0 otherwise. This matrix gives
            information about which rows in the memory matrix are valid.
        :param int memory_steps: number of steps to use for the memory, defaults to 0
        :param seed: seed to use for the random generator (None to ), defaults to None
        :param bool deterministic: if true it will always choose the highest probability action, otherwise it will use a weighted random
        """
        super().__init__()
        self.weight_funcs = weight_funcs

        self.memory = torch.zeros(memory_steps + 1, len(weight_funcs)) # first row is always 1, the rest shift and update
        self.memory[0,:] = 1

        self.steps = torch.zeros(memory_steps + 1, 1)
        self.steps[0,:] = 1

        self.seed = seed
        self.deterministic = deterministic
        self._w = torch.as_tensor(0, dtype=torch.float)
        self._p = torch.as_tensor(0, dtype=torch.float)

    @property
    def w(self) -> torch.Tensor:
        return self._w

    @property
    def p(self) -> torch.Tensor:
        return self._p

    @overrides(check_signature=False)
    def decide(self, pred:torch.Tensor, **kwargs) -> int:
        # obtain weights for labels
        self._w = torch.as_tensor([f(self.memory, self.steps) for f in self.weight_funcs], dtype=torch.float)
        # obtain scores and convert them to probabilities using softmax
        self._p = F.softmax((pred * self.w[None, :]).sum(1), dim=0) # (actions, labels) -> (actions)
        # choose the action to take
        if self.deterministic:
            res = self.p.argmax().item()
        else:
            res = default_rng(self.seed).choice(pred.shape[0], size=1, p=self.p.numpy()).item()
        
        # update memory
        if self.memory.shape[0] > 1:
            self.memory[2:,...] = self.memory[1:-1,...]
            self.memory[1,...] = pred[res,...]

            self.steps = self.steps.roll(shifts=1, dims=0)
            self.steps[0,...] = 1

        return res


if __name__ == '__main__':
    from game.simulator import GraphSimulator, load_simulator_yarn

    # simulator
    simulator = load_simulator_yarn()

    # agent
    # agent = RandomAgent(4444)
    agent = LabelDecisorAgent(
        label_predictor=TorchLabelPredictor(
            model_path=Path('./notebooks/saved_bert/200_[20]_[30]_1_False_bert-base-multilingual-cased_0.001_adamw_8_max_val_mcc_False_False_100'),
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
