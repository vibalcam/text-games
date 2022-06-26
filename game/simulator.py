import glob
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Tuple, Set, Optional
from tqdm.auto import tqdm

import dominate
import graphviz
import networkx as nx
from dominate.tags import *
from overrides import EnforceOverrides, overrides

from helper.helper import save_pickle, load_pickle
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
    def read(self) -> Tuple[str, List, Dict]:
        """Returns the text, state, and the choices available and the extras of the last decision taken (if any)"""
        pass

    @abstractmethod
    def transition(self, choice: Union[int, str]) -> Tuple[str, List, Dict]:
        """
        Transitions the story with the given choice

        :param choice: choice taken
        """
        pass

    @abstractmethod
    def get_current_path(self) -> List[str]:
        """Get the current path taken (history of choices taken)"""
        pass

    @abstractmethod
    def get_current_node_attr(self) -> Dict:
        """Get the current node attributes"""
        pass

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

    :param graph: graph that contains the game to run. It must at least have the following attributes:
        - Node: `text: str, attr: Dict, title:str`
        - Edge: `action: str, extras: Dict`
    """
    # node attributes
    ATTR_TEXT = 'text'
    ATTR_TITLE = 'title'
    ATTR_ATTR = 'attr'
    # edge attributes
    ATTR_ACTION = 'action'
    ATTR_EXTRAS = 'extras'
    # extras attributes
    ATTR_PRED = 'pred'

    def __init__(self, graph: nx.DiGraph):
        super().__init__()
        self.graph = graph
        # get root node
        self.root = [n for n, d in graph.in_degree() if d == 0][0]

        # initialize simulator, variables initialized in restart
        self.last_extra = None
        self.choices_history = None
        self.current = None
        self.actions = None
        self.showed_actions = None
        self.restart()

    @overrides(check_signature=False)
    def restart(self):
        self.current = self.root
        self.choices_history = []
        self.last_extra = {}
        self._get_actions()

    @overrides(check_signature=False)
    def read(self) -> Tuple[str, List, Dict]:
        # get node data
        data = self.graph.nodes(data=True)[self.current]
        # get possible actions
        self._get_actions()

        return data[self.ATTR_TEXT], self.showed_actions, self.last_extra

    @overrides(check_signature=False)
    def transition(self, choice: Union[int, str]) -> Tuple[str, List, Dict]:
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
        self.actions = {v[self.ATTR_ACTION]: {'node': k, self.ATTR_EXTRAS: v[self.ATTR_EXTRAS]}
                        for k, v in self.graph[self.current].items()}
        self.showed_actions = list(self.actions.keys())

    @overrides(check_signature=False)
    def get_current_path(self) -> List[str]:
        return self.choices_history

    @overrides(check_signature=False)
    def get_current_node_attr(self) -> Dict:
        n = self.graph.nodes(data=True)[self.current]
        d = n[self.ATTR_ATTR].copy()
        d[self.ATTR_TITLE] = n[self.ATTR_TITLE]
        return d

    @overrides(check_signature=False)
    def is_finished(self) -> bool:
        return self.graph.out_degree[self.current] == 0

    def create_html(self, folder_path: str):
        """Creates the html files necessary to run in the browser. Restarts the simulator."""
        self.restart()
        create_graph_html(self.graph, folder_path)

    def get_graphviz(self):
        dot = graphviz.Digraph()
        for n, attr in self.graph.nodes(data=True):
            dot.node(n, label=attr[self.ATTR_TITLE])
        for p, n, attr in self.graph.edges(data=True):
            dot.edge(p, n, label=attr[self.ATTR_ACTION])
        return dot


def load_simulator_yarn(
        yarn: str = 'yarnScripts',
        force: bool = False,
        graph_file_sfx: str = '_graph.pickle',
        **kwargs,
) -> GraphSimulator:
    """
    Loads a `GraphSimulator` from yarn files, checking if there is a graph saved already

    :param yarn: directory where the yarnScripts are saved
    :param force: whether to force the loading of the yarn files instead of trying to load a saved graph
    :param graph_file_sfx: suffix of the graph file

    :return: a `GraphSimulator` of the yarn files
    """
    yarn = Path(yarn)
    graph_f = yarn.parent.joinpath(f"{yarn.name}{graph_file_sfx}")
    if not force and graph_f.exists():
        graph = load_pickle(graph_f)
    else:
        graph = YarnSimulator(yarn=yarn, file_type='yarn', **kwargs).get_decision_graph(simplified=False)[0]
    
    save_pickle(graph, graph_f)
    return GraphSimulator(graph)


class YarnSimulator(Simulator):
    def __init__(
            self,
            yarn: Union[str, List[Dict]] = 'yarnScripts',
            file_type: str = 'yarn',
            jump_as_choice=True,
            text_unk_macro: Optional[str] = "",
            extras_separator: str = '__',
            extras_separator_key: str = ':'
    ):
        """
        Simulator
        :param yarn: path for the yarn files or the content of the yarn file (title, body...)
        :param file_type: file type of yarn. Either json, yarn (.txt, .yarn) or content
        :param jump_as_choice: whether to treat jumps as a choice with only one option (true) or as goto (false)
        :param text_unk_macro: if not None, placeholder when an unknown macro is encountered
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

        if file_type == 'yarn':
            self.data = get_content_from_yarn(yarn)
        elif file_type == 'json':
            with open(yarn, 'r') as file:
                self.data = json.load(file)
        elif file_type == 'content':
            self.data = yarn
        else:
            raise Exception("Type of content not allowed. Must be yarn, json or content")

        # defined in restart
        self.last_extras = None
        self.last_choices_list = None
        self.choices = None
        self.choices_history = None
        self.restart()

    @overrides(check_signature=False)
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

    def _parse_extras_dict(self, s: str) -> Dict:
        return dict([k.split(self.extras_separator_key) for k in s.split(self.extras_separator)])

    @overrides(check_signature=False)
    def read(self) -> Tuple[str, List, Dict]:
        # create dict of choices and tuple of (choice full name, extras as dictionary of key values)
        self.choices = {k[0]: (self.extras_separator.join(k),
                               self._parse_extras_dict(k[1].strip()) if len(k) > 1 else {})
                        for k in [c.split(self.extras_separator, 1) for c in self.controller.choices()]}

        # create list of choices in order to assure order and use int for transition
        self.last_choices_list = list(self.choices.keys())

        return self.controller.message(), self.last_choices_list, self.last_extras

    @overrides(check_signature=False)
    def get_current_node_attr(self) -> Dict:
        return self.controller.state.attr if self.controller.state.attr is not None else {}

    @overrides(check_signature=False)
    def transition(self, choice: Union[int, str]) -> Tuple[str, List, Dict]:
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

    @overrides(check_signature=False)
    def get_current_path(self) -> List[str]:
        return self.choices_history

    @overrides(check_signature=False)
    def is_finished(self) -> bool:
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
            return self._dfs(nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)

        if self._graph_complete is None:
            self._graph_complete = self._dfs(nx.DiGraph(), graphviz.Digraph(), set(), [], None, simplified)
        return self._graph_complete

    def create_html(self, folder_path: str):
        """Creates the html files necessary to run in the browser. Restarts the simulator."""
        self.restart()
        graph, _ = self.get_decision_graph(simplified=False)
        create_graph_html(graph, folder_path)

    def _dfs(
            self,
            graph: nx.DiGraph = nx.DiGraph(),
            dot: graphviz.Digraph = graphviz.Digraph(),
            nodes_visited=None,
            prev_path=None,
            parent_node_title=None,
            simplified=False,
    ) -> Tuple[nx.DiGraph, graphviz.Digraph]:
        """
        Deep First Search algorithm implementation.

        Generates a graph from the given simulator adding certain node and edge attributes:
        - Node: `text, state, attr, title`
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
        # default arguments
        if nodes_visited is None:
            nodes_visited = set()
        if prev_path is None:
            prev_path = []

        (text, cur_actions, extras) = self.read()
        curr_node = self.controller.state
        curr_node_title_vars = f"{curr_node.title}_{self.controller.get_game_locals()}".replace(":", '=')
        curr_node_title = curr_node.title if parent_node_title is None or simplified else curr_node_title_vars

        # Add node and edge
        if curr_node_title not in graph.nodes:
            graph.add_node(curr_node_title, text=text, attr=self.get_current_node_attr(), title=curr_node.title)
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
            if self.get_current_path() != prev_path:
                self.restart()
                self.read()
                for act in prev_path:
                    self.transition(act)

            self.transition(choice)
            self._dfs(graph, dot, nodes_visited, self.get_current_path().copy(), curr_node_title,
                      simplified=simplified)

        return graph, dot


def create_graph_html(graph: nx.DiGraph, folder_path: str, name: str = 'Game'):
    """
    Creates a html folder with the html files required to play the game
    :param folder_path: path where the folder is going to be created
    :param graph: complete decision graph. Starting node must be named 'Start'
    :param name: game name used as title for the html
    """
    html_folder = 'html'
    full_path = f'{folder_path}/{html_folder}'
    Path(full_path).mkdir(parents=True, exist_ok=True)

    for node in tqdm(graph.nodes):
        title = graph.nodes[node]['title']
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

    d = dominate.document(title=name)
    d += a('Start the Game', href=f"{html_folder}/start.html")
    with open(f"{folder_path}/index.html", "w") as file:
        file.write(d.render())


# def create_bondage_html(simulator: YarnSimulator, file_name: str):
#     """
#         todo FIX: DOES NOT CORRECTLY SHOW TEXT AFTER OPTIONS ARE CHOSE
#         Creates the html file to play the game. THE JS FOLDER MUST BE ALSO INCLUDED.
#         It uses the bondage (https://github.com/hylyh/bondage.js), a javascript yarn parser
#         :param simulator: YarnSimulator from which to create html
#         :param file_name: name of the html file created
#         """
#     d = dominate.document()
#     with d.head:
#         meta(charset="utf-8")
#         script(type='text/javascript', src='js/bondage.min.js')
#         script(type='text/javascript', src='js/run.js')
#
#     with d.body:
#         textarea(json.dumps(simulator.data, ensure_ascii=True), id='yarn', style='display:none')
#         div(id='display-area')
#
#     with open(file_name, "w") as file:
#         file.write(d.render())


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


# def yarn_to_json(simulator: YarnSimulator, json_file: str):
#     with open(json_file, 'w', encoding='utf-8') as f:
#         json.dump(simulator.data, f, ensure_ascii=True, indent=4)


if __name__ == '__main__':
    # os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'

    # Load and get html
    s = load_simulator_yarn(yarn='./yarnScripts', text_unk_macro="", jump_as_choice=True)
    s.create_html('./examples/')
    
    
    # Graph in multiple pages
    # s.get_graphviz().save(filename='dot', directory='./examples/')
    # dot -Gpage="8,12" -Gsize="220,6" -Tps -o dot.ps  dot
    # s.get_graphviz().render(filename='graph_decision', directory='./examples/', view=True,
    #                         cleanup=True, format='svg')

    # Narrative graph
    # s2 = YarnSimulator(text_unk_macro="", yarn='./yarnScripts', jump_as_choice=True)
    # s2.get_decision_graph(simplified=True)[1].render(filename='graph_narrative', directory='./examples/', view=True,
    #                                                  cleanup=True, format='svg')

    print("Done")


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
