import glob
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Union, Tuple, Set

from overrides import EnforceOverrides, overrides
import dominate
import graphviz
import networkx as nx
from dominate.tags import *
from models.utils import load_pickle, save_pickle

from yarnPygame.src.yarn import YarnController


class Simulator(ABC, EnforceOverrides):
    """
    Interface for simulators
    """

    @abstractmethod
    def restart(self):
        """Resets the simulator"""
        pass

    @abstractmethod
    def read(self) -> Tuple[str, List, Dict[str,str]]:
        """Returns the text, state, and the choices available and the extras (if any)"""
        pass

    @abstractmethod
    def transition(self, choice: Union[int,str]) -> Tuple[str, List, Dict[str,str]]:
        """
        Transitions the story with the given choice
        :param choice: choice taken
        """
        pass

    @abstractmethod
    def get_current_path(self) -> List[str]:
        """Get the current path taken (history of choices taken)"""
        raise NotImplementedError()

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Returns true if the simulation has finished
        :return: bool
        """
        pass


class GraphSimulator(Simulator):
    """
    Simulator that uses a graph to run

    :param graph: graph that contains the game to run
    """
    ATTR_TEXT = 'text'
    ATTR_ACTION = 'action'
    ATTR_EXTRAS = 'extras'

    def __init__(self, graph:nx.DiGraph):
        super().__init__()
        self.graph = graph
        # get root node
        self.root = [n for n,d in graph.in_degree() if d==0][0]
        # initialize simulator
        self.restart()

    @overrides
    def restart(self):
        self.current = self.root
        self.choices_history = []
        self.last_extra = {}
        self._get_actions()

    @overrides
    def read(self) -> Tuple[str, List, Dict[str,str]]:
        # get node data
        data = self.graph.nodes(data=True)[self.current]
        # get possible actions
        self._get_actions()

        return data[self.ATTR_TEXT], self.showed_actions, self.last_extra

    @overrides
    def transition(self, choice: Union[int,str]) -> Tuple[str, List, Dict[str,str]]:
        # if given as int transform to str
        if type(choice) == int:
            choice = self.showed_actions[choice]
        
        # get next node and extra of action
        self.current = self.actions[choice]['node']
        self.last_extra = self.actions[choice][self.ATTR_EXTRAS]
        # save choice taken
        self.choices_history.append(choice)
        
        return self.read()

    def _get_actions(self):
        self.actions = {v[self.ATTR_ACTION]:{'node':k, self.ATTR_EXTRAS:v[self.ATTR_EXTRAS]} 
                        for k,v in self.graph[self.current].items()}
        self.showed_actions = list(self.actions.keys())

    @overrides
    def get_current_path(self) -> List[str]:
        return self.choices_history

    @overrides
    def is_finished(self) -> bool:
        self.graph.out_degree[self.current] == 0


# todo finish this
def load_simulator_yarn(
    yarn: Union[str, List[Dict]] = 'yarnScripts',
    force:bool = False,
    graph_file_sfx = '_graph.pickle', 
    **kwargs
) -> GraphSimulator:
    """
    Loads a `GraphSimulator` from yarn files, checking if there is a graph saved already

    :param yarn: 
    :param force:
    :param graph_file_sfx:

    :return: a `GraphSimulator` of the yarn files
    """
    graph_file = yarn + graph_file_sfx
    if isinstance(yarn,str) and not force:
        graph = load_pickle(graph_file)
    else:
        graph = None
        save_pickle(graph, graph_file)
    
    return GraphSimulator(graph)


class YarnSimulator(Simulator):
    def __init__(self, yarn: Union[str, List[Dict]] = 'yarnScripts', file_type: str = 'yarn', jump_as_choice=True,
                 text_unk_macro=None, extras_separator: str = '__', extras_separator_key:str = ':'):
        """
        Simulator
        :param yarn: path for the yarn files or the content of the yarn file (title, body...)
        :param type: file type of yarn. Either json, yarn (.txt, .yarn) or content
        :param jump_as_choice: whether to treat jumps as a choice with only one option (true) or as goto (false)
        :param text_unk_macro: placeholder when an unknown macro is encountered
        :param extras_separator: token used to separate the extras from the text (ex. prueba__r:2, means it has reward 2)
        :param extras_separator_key: token used to separate the key and value in extras
        """
        self.jump_as_choice = jump_as_choice
        self.text_unk_macro = text_unk_macro
        self._graph_complete = None
        self.extras_separator = extras_separator
        self.extras_separator_key = extras_separator_key

        # Initialized in self.restart()
        self._controller = None

        # # The choices made up until this moment
        # self.choices_history = []
        # # Dictionary with choices as keys and extras as values
        # self.choices = {}
        # # Last extras obtained
        # self.last_extras = ""

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

    @overrides
    def restart(self):
        self._controller = YarnController(None, False, content=self.data,
                                         jump_as_choice=self.jump_as_choice,
                                         text_unk_macro=self.text_unk_macro)
        # list of choices taken
        self.choices_history = []
        # choices with their respective information
        self.choices = {}
        # list of choices presented
        self.last_choices_list = []
        # last extras received
        self.last_extras = {}

    def _parse_extras_dict(self, s:str) -> Dict[str, str]:
        return dict([k.split(self.extras_separator_key) for k in s.split(self.extras_separator)])

    @overrides
    def read(self) -> Tuple[str, List, Dict[str,str]]:
        # create dict of choices and tuple of (choice full name, extras as dictionary of key values)
        self.choices = {k[0]: (self.extras_separator.join(k), 
                            self._parse_extras_dict(k[1].strip()) if len(k) > 1 else {}) 
                        for k in [c.split(self.extras_separator, 1) for c in self.controller.choices()]}
        
        # create list of choices in order to assure order and use int for transition
        self.last_choices_list = list(self.choices.keys())

        return self.controller.message(), self.last_choices_list, self.last_extras

    # todo get node extras to graph also

    # todo get current node attributes

    @overrides
    def transition(self, choice: Union[int, str]) -> Tuple[str, List, Dict[str,str]]:
        """
        Transitions the story with the given choice
        :param choice: choice taken
        :return: state, choices and extras (if any) after the transition
        """
        # from int to str
        if type(choice) == int:
            choice = self.last_choices_list[choice]

        self.choices_history.append(choice)
        choice_dict = self.choices[choice]
        self.last_extras = choice_dict[1]
        self.controller.transition(choice_dict[0])
            # choice if not self.last_extras else f"{choice}{self.extras_separator}{self.last_extras}")
        return self.read()

    @property
    def controller(self):
        return self._controller

    @overrides
    def get_current_path(self) -> List[str]:
        return self.choices_history

    @overrides
    def is_finished(self):
        return self.controller.finished

    def get_decision_graph(self, simplified=False) -> Tuple[nx.DiGraph, graphviz.Digraph]:
        """
        Returns the decision graph. Restarts the simulator.
        :param simplified: if true it returns the narrative of the story (the story line),
                            if false it returns the complete decision graph
        :return: a tuple containing a networkx graph (nodes attr: `state`, `text`, `extras`; edges attr: `action`)
                and a pydot graph
        """
        self.restart()
        if simplified:
            return dfs(self, nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)

        if self._graph_complete is None:
            self._graph_complete = dfs(self, nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)
        return self._graph_complete

    # todo no bondage since does not work
    def create_html(self, folder_path: str, use_bondage=False):
        """Creates the html files necessary to run in the browser. Restarts the simulator."""
        self.restart()
        graph, _ = self.get_decision_graph(simplified=False)
        if use_bondage:
            create_bondage_html(self, f'{folder_path}/yarn.html')
        else:
            create_graph_html(graph, folder_path)

# todo quitar
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


# todo fix this
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

# todo set as private method
def dfs(
    simulator: YarnSimulator, 
    graph: nx.DiGraph = nx.DiGraph(), 
    dot: graphviz.Digraph = graphviz.Digraph(), 
    nodes_visited: Set = set(), 
    prev_path: List = [],
    parent_node_title = None, 
    simplified=False,
) -> Tuple[nx.DiGraph, graphviz.Digraph]:
    """
    Deep First Search algorithm implementation.

    Generates a graph from the given simulator adding certain node and edge attributes:
    - Node: `text, state`
    - Edge: `action`

    :param simulator: simulator that runs the game
    :param graph: networkx graph where the game is recorded
    :param dot: graphviz graph that can be used to easily render the graph
    :param nodes_visited: set of visited nodes
    :param prev_path: list of the actions taken up until this point
    :param parent_node_title: title of the parent node
    :param simplified: if true it uses as node id the node's title (the story line),
                        if false it uses just the title with variables (complete decision graph)
    """
    # todo make generic for simulator
    (text, cur_actions, extras) = simulator.read()
    curr_node = simulator.controller.state
    curr_node_title_vars = f"{curr_node.title}_{simulator.controller.get_game_locals()}".replace(":", '=')
    curr_node_title = curr_node.title if parent_node_title is None or simplified else curr_node_title_vars

    # Add node and edge
    if curr_node_title not in graph.nodes:
        graph.add_node(curr_node_title, state=curr_node, text=text)
        dot.node(curr_node_title, label=curr_node.title)
    if parent_node_title is not None and (parent_node_title, curr_node_title) not in graph.edges:
        graph.add_edge(parent_node_title, curr_node_title, action=prev_path[-1], extras=extras)
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
                simulator.transition(act)
                # (_, actions, _) = simulator.read()

        # (_, actions, _) = simulator.read()
        # curr_path.append(choice)
        simulator.transition(choice)
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
    # sim.get_decision_graph(simplified=True)[1].render(filename='graph_narrative', view=True,
    #                                                   cleanup=True, format='svg')
    # sim.create_html('.')


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
