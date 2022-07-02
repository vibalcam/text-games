from typing import List, Dict, Tuple
import pickle

import networkx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef, mean_squared_error
import networkx as nx

from game.simulator import GraphSimulator, YarnSimulator, Simulator


# Labels for the actions
from helper.helper import save_pickle

LABELS = ['bad', 'good']


class StateActionDataset(Dataset):
    """
    Class that represents a dataset to train the classifier
    """

    def __init__(
            self,
            states: List[str],
            actions: List[str],
            rewards: List[float],
            tokenize: bool = True,
            tokenizer=None,
            max_length: int = None,
            device=None,
            balanced_actions:bool = False,
            augment_negative:bool = False,
    ):
        """
        Initializer for the dataset
        :param states: list of states (L)
        :param actions: list of actions (L)
        :param rewards: list of rewards (L) (0 or 1)
        :param tokenize: whether to tokenize the input
        :param tokenizer: tokenizer. Only used if tokenize is true.
        :param max_length: max length for tokenization. Only used if tokenize is true
        :param device: the desired device of returned data. Only used if tokenize is true
        :param balanced_actions: if true, it will ensure that the actions returned are balanced
        :param augment_negative: if true, it will add a negated version of all the samples by adding "No" in front of all actions
        """
        # Check correct parameters
        if tokenize and tokenizer is None:
            raise Exception("Tokenizer is mandatory if tokenize true")

        self.device = device
        self.tokenize = tokenize

        # augment with the negative version of the actions by adding "No"
        if augment_negative:
            actions.extend([f"No {k[0].lower()}{k[1:]}" for k in actions])
            states.extend(states.copy())
            rewards.extend([1-k for k in rewards])

        self.balanced_actions = balanced_actions
        if balanced_actions:
            s_actions = pd.Series(actions)
            self.act_idx = [np.nonzero((s_actions == k).to_numpy())[0] for k in s_actions.unique()]
            # mask has all actions by index of a certain type
            # where returns indices where it is true
            # choose randomly one of those indices
            self.act_perm = [np.random.permutation(m).tolist() for m in self.act_idx]

        self.rewards = torch.as_tensor(rewards, dtype=torch.float)
        # if no tokenize, save lists
        if not tokenize:
            self.states = states
            self.actions = actions
            return

        # If none is given for max_length or it exceeds maximum, use max length from tokenizer
        if max_length is None or max_length > tokenizer.model_max_length:
            max_length = tokenizer.model_max_length

        # Truncate states and actions to max length and make sure not to cut a phrase in half
        self.states = [k[-max_length:].split('\n', 1)[-1] for k in states]
        self.actions = [k[-max_length:].split('\n', 1)[-1] for k in actions]

        # Tokenize states and actions with padding
        self.states = tokenizer(self.states, return_tensors='pt',
                                padding='max_length', max_length=max_length)  # (L) -> dict{k, (L, input_size)}
        self.actions = tokenizer(self.actions, return_tensors='pt',
                                 padding='max_length', max_length=max_length)  # (L) -> dict{k, (L, input_size)}

    def __len__(self):
        """
        :return: the length of the dataset
        """
        return self.rewards.shape[0] if not self.balanced_actions else len(self.act_idx)

    def _get_item_balanced(self, idx):
        act_type = self.act_perm[idx]
        if not act_type:
            act_type = np.random.permutation(self.act_idx[idx]).tolist()
            self.act_perm[idx] = act_type

        return act_type.pop()

    def __getitem__(self, idx):
        """
        Returns an item from the dataset given the index
        :param idx: index to retrieve
        :return: item retrieved (state: str, action:str, reward: float)
        """
        if self.balanced_actions:
            idx = self._get_item_balanced(idx)

        if not self.tokenize:
            return self.states[idx], self.actions[idx], self.rewards[idx]

        if self.device is None:
            return {k: (v[idx, ...]) for k, v in self.states.items()}, \
                   {k: (v[idx, ...]) for k, v in self.actions.items()}, \
                   self.rewards[idx]
        else:
            return {k: (v[idx, ...].to(self.device)) for k, v in self.states.items()}, \
                   {k: (v[idx, ...].to(self.device)) for k, v in self.actions.items()}, \
                   self.rewards[idx].to(self.device)


def load_data(
    graph: networkx.DiGraph,
    num_workers=0,
    batch_size=4,
    drop_last=False,
    lengths=(0.75, 0.15, 0.1),
    random_seed: int = 4444,
    reward_key:str='r',
    balanced_actions:bool = False,
    balanced_actions_test:bool = False,
    augment_negative:bool = False,
    **kwargs
) -> Tuple[DataLoader, ...]:
    """
    Method used to load the dataset. It retrives the data with random shuffle

    :param graph: graph for the game. It must at least have the following attributes:
        - Node: `text,`
        - Edge: `action,extras`
    :param random_seed: seed to randomly shuffle data
    :param num_workers: how many subprocesses to use for data loading.
                        0 means that the data will be loaded in the main process
    :param batch_size: size of each batch which is retrieved by the dataloader
    :param drop_last: whether to drop the last batch if it is smaller than batch_size
    :param lengths: tuple with percentage of train, validation and test samples
    :param reward_key: attribute key for the reward
    :param balanced_actions: if true, it will ensure that, during training, the actions returned are balanced
    :param balanced_actions_test: if true, it will ensure that, during validation and testing, the actions returned are balanced
    :param augment_negative: if true, it will add a negated version of all the samples by adding "No" in front of all actions

    :return: tuple of dataloader (same length as parameter lengths)
    """

    # Get data from simulator
    # graph = YarnSimulator(dataset_path, jump_as_choice=True, text_unk_macro="").get_decision_graph()[0]

    states = []
    actions = []
    rewards = []
    for p, _, attr in graph.edges(data=True):
        if reward_key not in attr[GraphSimulator.ATTR_EXTRAS]:
            continue
        states.append(graph.nodes[p][GraphSimulator.ATTR_TEXT].strip())
        actions.append(attr[GraphSimulator.ATTR_ACTION].strip())
        rewards.append(float(attr[GraphSimulator.ATTR_EXTRAS][reward_key]))

    # Shuffle the data
    data = list(zip(states, actions, rewards))
    np.random.default_rng(random_seed).shuffle(data)
    states, actions, rewards = [list(k) for k in zip(*data)]

    # Get datasets
    lengths = [int(k * len(states)) for k in lengths[:-1]]
    lengths = np.cumsum(lengths)

    datasets = [StateActionDataset(states[:lengths[0]],
                                    actions[:lengths[0]],
                                    rewards[:lengths[0]],
                                    balanced_actions=balanced_actions,
                                    augment_negative=augment_negative,
                                    **kwargs)]
    datasets.extend([StateActionDataset(states[lengths[k]:lengths[k + 1]],
                                        actions[lengths[k]:lengths[k + 1]],
                                        rewards[lengths[k]:lengths[k + 1]],
                                        balanced_actions=balanced_actions_test,
                                        **kwargs) for k in range(len(lengths) - 1)])
    datasets.append(StateActionDataset(states[lengths[-1]:],
                                        actions[lengths[-1]:],
                                        rewards[lengths[-1]:],
                                        balanced_actions=balanced_actions_test,
                                        **kwargs))

    # Return DataLoaders for the datasets
    return tuple(DataLoader(
        k,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=idx == 0,
        drop_last=drop_last,
    ) for idx, k in enumerate(datasets))


def get_certainty(pred: np.array) -> np.array:
    """
    Calculates the certainty of the decisions given the predictions

    :param np.array pred: the predictions
    :return np.array: the certainty of the decisions
    """
    chosen = (pred > 0).astype(float)
    return (np.abs(chosen-1+pred) * 100)


class ConfusionMatrix:
    """
    Class that represents a confusion matrix. 
    
    Cij is equal to the number of observations known to be in class i and predicted in class j
    """

    def _make(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Returns the confusion matrix of the given predicted and labels values
        :param preds: predicted values (B)
        :param labels: true values (B)
        :return: (size,size) confusion matrix of `size` classes
        """
        matrix = torch.zeros(self.size, self.size, dtype=torch.float)
        for t, p in zip(labels.reshape(-1).long(), preds.reshape(-1).long()):
            matrix[t, p] += 1
        return matrix

    def __init__(self, size, name:str=''):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        :param name: name of the confusion matrix
        """
        self.matrix = torch.zeros(size, size, dtype=torch.float)
        self.preds = None
        self.labels = None
        self.name = name

    def __repr__(self) -> str:
        return self.matrix.numpy().__repr__()

    def add(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        :param preds: predicted values (B)
        :param labels: true values (B)
        """
        preds = preds.reshape(-1).float().cpu().detach().clone()
        labels = labels.reshape(-1).float().cpu().detach().clone()
        
        self.matrix += self._make(preds, labels)
        self.preds = torch.cat((self.preds, preds), dim=0) if self.preds is not None else preds
        self.labels = torch.cat((self.labels, labels), dim=0) if self.labels is not None else labels

    @property
    def size(self):
        return self.matrix.shape[0]

    @property
    def matthews_corrcoef(self):
        """
        Matthews correlation coefficient (MCC) 
        https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
        """
        return matthews_corrcoef(y_true=self.labels.numpy(),y_pred=self.preds.numpy())

    @property
    def rmse(self):
        return mean_squared_error(y_true=self.labels.numpy(),y_pred=self.preds.numpy(), squared=False)

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean().item()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)

    @property
    def normalize(self):
        return self.matrix / (self.matrix.sum() + 1e-5)

    def visualize(self, normalize: bool = False):
        """
        Visualize confusion matrix
        :param normalize: whether to normalize the matrix by the total amount of samples
        """
        plt.figure(figsize=(15, 10))

        matrix = self.normalize.numpy() if normalize else self.matrix.numpy()

        df_cm = pd.DataFrame(matrix).astype(int)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return plt


class Accuracy:
    """
    Calculates the accuracy of the predictions
    """

    def __init__(self) -> None:
        self.correct = np.zeros(1, dtype=np.float32)
        self.samples = np.zeros(1, dtype=np.float32)

    def add(self, predicted: torch.Tensor, label: torch.Tensor):
        """
        Adds samples for the accuracy calculation
        It considers predicted to be class 1 if probability is higher than 0.5
        :param predicted: the input prediction
        :param label: the real label
        """
        self.samples += predicted.numel()
        correct = ((predicted > 0).float() == label).float().sum().cpu().detach().numpy()
        self.correct += correct

    @property
    def accuracy(self):
        return (self.correct / self.samples).item()


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    :param seed: seed for the random generators
    """
    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

        # for deterministic behavior on cuda >= 10.2
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# if __name__ == '__main__':
#     s = YarnSimulator(text_unk_macro="", yarn='../yarnScripts')
#     g = s.get_decision_graph[0]
#     print("Done")
