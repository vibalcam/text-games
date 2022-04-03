import pathlib
import pickle
from ast import literal_eval
from os import path
from typing import List, Dict, Union


class BiDict(dict):
    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        value = self[key]
        self.inverse.setdefault(value, []).remove(key)
        if value in self.inverse and not self.inverse[value]:
            del self.inverse[value]
        super(BiDict, self).__delitem__(key)

    def get_inverse(self, value) -> List:
        return self.inverse[value]

    def save(self, file_name: str):
        with open(file_name, 'w') as tasks_file:
            tasks_file.write(str(self))
        # with open(file_name, 'wb') as tasks_file:
        #     pickle.dump(dict(self), tasks_file)

    @staticmethod
    def load(file_name: str):
        if path.isfile(file_name):
            with open(file_name, 'r') as tasks_file:
                loaded = BiDict(literal_eval(tasks_file.read()))
            # with open(file_name, 'rb') as tasks_file:
            #     loaded = BiDict(pickle.load(tasks_file))
        else:
            raise Exception(f"{file_name} does not exist")
        return loaded


def save_pickle(obj, path: Union[str, pathlib.Path]):
    """
    Saves an object with pickle
    :param obj: object to be saved
    :param save_path: path to the file where it will be saved
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, pathlib.Path]):
    """
    Loads an object with pickle from a file
    :param path: path to the file where the object is stored
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_dict(d: Dict, path: Union[str, pathlib.Path]) -> None:
    """
    Saves a dictionary to a file in plain text
    :param d: dictionary to save
    :param path: path of the file where the dictionary will be saved
    """
    with open(path, 'w', encoding="utf-8") as file:
        file.write(str(d))


def load_dict(path: Union[str, pathlib.Path]) -> Dict:
    """
    Loads a dictionary from a file in plain text
    :param path: path where the dictionary was saved
    :return: the loaded dictionary
    """
    with open(path, 'r', encoding="utf-8") as file:
        from ast import literal_eval
        loaded = dict(literal_eval(file.read()))
    return loaded
