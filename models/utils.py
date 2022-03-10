from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from simulator import YarnSimulator

# Labels for the actions
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
    ):
        """
        Initializer for the dataset
        :param states: list of states (L)
        :param actions: list of actions (L)
        :param rewards: list of rewards (L)
        :param tokenize: whether to tokenize the input
        :param tokenizer: tokenizer. Only used if tokenize is true.
        :param max_length: max length for tokenization. Only used if tokenize is true
        :param device: the desired device of returned data. Only used if tokenize is true
        """
        # Check correct parameters
        if tokenize and tokenizer is None:
            raise Exception("Tokenizer is mandatory if tokenize true")

        self.device = device
        self.tokenize = tokenize
        self.rewards = torch.as_tensor(rewards, dtype=torch.float)

        # if no tokenize, save lists
        if not tokenize:
            self.states = states
            self.actions = actions
            return

        # todo do not cut, let tokenizer do the cutting

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
        return self.rewards.shape[0]

    def __getitem__(self, idx):
        """
        Returns an item from the dataset given the index
        :param idx: index to retrieve
        :return: item retrieved (state: str, action:str, reward: float)
        """
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


def load_data(dataset_path: str = '../yarnScripts', num_workers=0, batch_size=4, drop_last=False,
              lengths=(0.8, 0.1, 0.1), random_seed: int = 123, reward_key:str='r', **kwargs) -> Tuple[DataLoader, ...]:
    """
    Method used to load the dataset. It retrives the data with random shuffle
    :param dataset_path: path to the dataset
    :param random_seed: seed to randomly shuffle data
    :param num_workers: how many subprocesses to use for data loading.
                        0 means that the data will be loaded in the main process
    :param batch_size: size of each batch which is retrieved by the dataloader
    :param drop_last: whether to drop the last batch if it is smaller than batch_size
    :param lengths: tuple with percentage of train, validation and test samples
    :param reward_key: attribute key for the reward

    :return: tuple of dataloader (same length as parameter lengths)
    """

    # Get data from simulator
    graph = YarnSimulator(dataset_path, jump_as_choice=True, text_unk_macro="").get_decision_graph()[0]

    states = []
    actions = []
    rewards = []
    for p, _, attr in graph.edges(data=True):
        #todo change extras
        if not attr['extras']:
            continue
        states.append(graph.nodes[p]['text'].strip())
        actions.append(attr['action'])
        rewards.append(float(attr['extras'][reward_key]))

    # Shuffle the data
    data = list(zip(states, actions, rewards))
    np.random.default_rng(random_seed).shuffle(data)
    states, actions, rewards = [list(k) for k in zip(*data)]

    # Get datasets
    lengths = [int(k * len(states)) for k in lengths[:-1]]
    lengths = np.cumsum(lengths)

    datasets = [StateActionDataset(states[:lengths[0]], actions[:lengths[0]], rewards[:lengths[0]], **kwargs)]
    datasets.extend([StateActionDataset(states[lengths[k]:lengths[k + 1]], actions[lengths[k]:lengths[k + 1]],
                                        rewards[lengths[k]:lengths[k + 1]], **kwargs) for k in range(len(lengths) - 1)])
    datasets.append(StateActionDataset(states[lengths[-1]:], actions[lengths[-1]:], rewards[lengths[-1]:], **kwargs))

    # Return DataLoaders for the datasets
    return tuple(DataLoader(
        k,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=idx == 0,
        drop_last=drop_last,
    ) for idx, k in enumerate(datasets))


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
        return self.correct / self.samples


def save_dict(d: Dict, path: str) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    """
    with open(path, 'w', encoding="utf-8") as file:
        file.write(str(d))


def load_dict(path: str) -> Dict:
    """
    Loads a dictionary from a file in plain text
    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        loaded = dict(literal_eval(file.read()))
    return loaded


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_list(path: str) -> List:
    """
    Loads a list from a file in plain text
    :param path: path where the list was saved
    :return: the loaded list
    """
    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        loaded = list(literal_eval(file.read()))
    return loaded


if __name__ == '__main__':
    pass
# todo load dataset and check length
