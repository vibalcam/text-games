import os
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Set

import graphviz
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from helper import BiDict
from networkx.drawing.nx_pydot import graphviz_layout
import glob

from yarnPygame.src.yarn import YarnController

import json
from pathlib import Path
import dominate
from dominate.tags import *


class Simulator(ABC):
    """
    Interface for simulators
    """

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def read(self):
        """Returns the text state, the choices available and the reward (if any)"""
        pass

    @abstractmethod
    def act(self, choice):
        pass

    def get_current_path(self):
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self):
        pass


class YarnSimulator(Simulator):
    def __init__(self, yarn: Union[str, List[Dict]] = 'yarnScripts', file_type: str = 'yarn', jump_as_choice=True,
                 text_unk_macro=None):
        """
        Simulator
        :param yarn: path for the yarn files or the content of the yarn file (title, body...)
        :param type: file type of yarn. Either json, yarn (.txt, .yarn) or content
        :param jump_as_choice: whether to treat jumps as a choice with only one option (true) or as goto (false)
        :param text_unk_macro: placeholder when an unknown macro is encountered
        """
        self.jump_as_choice = jump_as_choice
        self.text_unk_macro = text_unk_macro
        self.graph_complete = None

        if file_type == 'yarn':
            self.data = get_content_from_yarn(yarn)
        elif file_type == 'json':
            with open(yarn, 'r') as file:
                self.data = json.load(file)
        elif file_type == 'content':
            self.data = yarn
        else:
            raise Exception("Type of content not allowed. Must be yarn, json or content")

        self.restart()

    def restart(self):
        self.controller = YarnController(None, False, content=self.data,
                                         jump_as_choice=self.jump_as_choice,
                                         text_unk_macro=self.text_unk_macro)

    def read(self) -> Tuple[str, List, float]:
        return self.controller.message(), self.controller.choices(), 0.0

    def act(self, choice: Union[int, str]):
        self.controller.transition(choice)
        return self.read()

    def get_current_path(self) -> List[str]:
        return self.controller.locals["path"]

    def is_finished(self):
        return self.controller.finished

    def get_decision_graph(self, simplified=False):
        """
        Returns the decision graph. Restarts the simulator.
        :param simplified: if true it returns the complete decision graph,
                            if false it returns the narrative of the story (the story line)
        :return: a tuple containing a networkx graph (nodes attr: state, text, reward; edges attr: action)
                and a pydot graph
        """
        self.restart()
        if simplified:
            return dfs(self, nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)

        if self.graph_complete is None:
            self.graph_complete = dfs(self, nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)
        return self.graph_complete

    def create_html(self, folder_path: str, use_bondage=False):
        """Creates the html files necessary to run in the browser. Restarts the simulator."""
        self.restart()
        graph, _ = self.get_decision_graph(simplified=False)
        if use_bondage:
            create_bondage_html(self, f'{folder_path}/yarn.html')
        else:
            create_graph_html(graph, folder_path)


def create_bondage_html(simulator: YarnSimulator, file_name: str):
    """
        todo FIX: DOES NOT CORRECTLY SHOW TEXT AFTER OPTIONS ARE CHOSE
        Creates the html file to play the game. THE JS FOLDER MUST BE ALSO INCLUDED.
        It uses the bondage (https://github.com/hylyh/bondage.js), a javascript yarn parser
        :param simulator: YarnSimulator from which to create html
        :param file_name: name of the html file created
        """
    d = dominate.document()
    with d.head:
        meta(charset="utf-8")
        script(type='text/javascript', src='js/bondage.min.js')
        script(type='text/javascript', src='js/run.js')

    with d.body:
        textarea(json.dumps(simulator.data, ensure_ascii=True), id='yarn', style='display:none')
        div(id='display-area')

    with open(file_name, "w") as file:
        file.write(d.render())


def create_graph_html(graph: nx.DiGraph, folder_path: str):
    """
    Creates a html folder with the html files required to play the game
    :param folder_path: path where the folder is going to be created
    :param graph: complete decision graph. Starting node must be named 'Start'
    """
    html_folder = 'html'
    full_path = f'{folder_path}/{html_folder}'
    Path(full_path).mkdir(parents=True, exist_ok=True)

    for node in graph.nodes:
        title = graph.nodes['Start']['state'].title
        text = graph.nodes[node]['text']

        # create page
        d = dominate.document(title=title)
        d += div(text, id='text', style="white-space: pre-line")
        with d.add(div(id='choices')).add(ul()):
            # create links
            for next, data_next in graph.adj[node].items():
                li(a(data_next['action'], href=f"{str(next).lower()}.html"))

        with open(f"{full_path}/{str(node).lower()}.html", "w") as file:
            file.write(d.render())

    d = dominate.document(title='index')
    d += a('Start the game', href=f"{html_folder}/start.html")
    with open(f"{folder_path}/index.html", "w") as file:
        file.write(d.render())


def dfs(simulator: YarnSimulator, graph: nx.DiGraph, dot: graphviz.Digraph, nodes_visited: Set, prev_path: List,
        parent_node_title, simplified=False):
    # todo make generic for simulator
    (text, cur_actions, reward) = simulator.read()
    curr_node = simulator.controller.state
    curr_node_title_vars = f"{curr_node.title}_{simulator.controller.get_game_locals()}".replace(":", '=')
    curr_node_title = curr_node.title if parent_node_title is None or simplified else curr_node_title_vars

    # Add node and edge
    if curr_node_title not in graph.nodes:
        graph.add_node(curr_node_title, state=curr_node, text=text, reward=reward)
        dot.node(curr_node_title, label=curr_node.title)
    if parent_node_title is not None and (parent_node_title, curr_node_title) not in graph.edges:
        graph.add_edge(parent_node_title, curr_node_title, action=prev_path[-1])
        dot.edge(parent_node_title, curr_node_title, label=prev_path[-1])

    # Check if already visited
    if curr_node_title_vars in nodes_visited:
        return
    nodes_visited.add(curr_node_title_vars)

    # Loop through options
    for choice in cur_actions:
        # move controller to the current state
        if simulator.get_current_path() != prev_path:
            simulator.restart()
            (_, actions, _) = simulator.read()
            # curr_path.clear()
            for act in prev_path:
                # curr_path.append(act)
                simulator.controller.transition(act)
                # (_, actions, _) = simulator.read()

        # (_, actions, _) = simulator.read()
        # curr_path.append(choice)
        simulator.controller.transition(choice)
        dfs(simulator, graph, dot, nodes_visited, simulator.get_current_path().copy(), curr_node_title,
            simplified=simplified)

    return graph, dot


def get_content_from_yarn(path: str) -> List[Dict]:
    """
    Gets the content from the yarn files found recursively in the path
    :param path: path to search for yarn files (.txt or .yarn)
    :return: the contents as a list of dictionary
    """
    files = glob.glob(f'{path}/**/*.txt', recursive=True)
    files.extend(glob.glob(f'{path}/**/*.yarn', recursive=True))

    data = []
    for path in files:
        node = {}
        with open(path, 'r', encoding="utf8") as file:
            line = file.readline()
            while "---" not in line:
                p = line.split(':')
                node[p[0].strip()] = p[1].strip()
                line = file.readline()
            node["body"] = file.read()
        data.append(node)

    return data


def yarn_to_json(simulator: YarnSimulator, json_file: str):
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(simulator.data, f, ensure_ascii=True, indent=4)


if __name__ == '__main__':
    sim = YarnSimulator()
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
    sim.get_decision_graph(simplified=False)[1].render(filename='graph_decision', view=True,
                                                       cleanup=True, format='svg')
    sim.get_decision_graph(simplified=True)[1].render(filename='graph_narrative', view=True,
                                                      cleanup=True, format='svg')
    sim.create_html('.', use_bondage=False)
    sim.create_html('.', use_bondage=True)

    # shortest_paths = BiDict(nx.single_source_shortest_path_length(graph,'Final'))
    #
    # for k,l in shortest_paths.inverse.items():
    #     c = graphviz.Digraph()
    #     c.attr(rank='same')
    #     for n in l:
    #         c.node(n)
    #     dot.subgraph(c)

    # http://www.graphviz.org/docs/attrs/rank/
    #
    # # graphviz
    # dot.attr(ranksep="3")
    # dot.render(view=True, cleanup=True)
    #
    # # networkx
    # pos = graphviz_layout(graph, prog="dot")
    # plt.figure(figsize=(12,12))
    # nx.draw_networkx(graph, pos)
    # plt.show()
    #
    # plt.savefig("g1.png", format="PNG")
