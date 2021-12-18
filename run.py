import os

import graphviz
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from helper import BiDict
from networkx.drawing.nx_pydot import graphviz_layout
import glob

from src.simulator import Simulator, YarnSimulator
from yarn import controller

import json
from pathlib import Path
import dominate
from dominate.tags import *

# name = "prueba"
#
# files = glob.glob('yarnScripts/**/*.txt', recursive=True)
# files.extend(glob.glob('yarnScripts/**/*.yarn', recursive=True))
# # files = ["Day1.yarn.txt", "Day2.yarn.txt"]
#
# data = []
# for path in files:
#     node = {}
#     with open(path, 'r', encoding="utf8") as file:
#         line = file.readline()
#         while "---" not in line:
#             p = line.split(':')
#             node[p[0].strip()] = p[1].strip()
#             line = file.readline()
#         node["body"] = file.read()
#     data.append(node)
#
# controller=controller.YarnController(None, name, False, content=data)
#
# while not controller.finished:
#     print(controller.message())
#     print(controller.transition(0))
#
# print("Done")


simulator = YarnSimulator()


# DFS algorithm
nodes_visited = set()
curr_path = []
graph = nx.DiGraph()
dot = graphviz.Digraph()


def dfs(simulator, prev_path, parent_node_title):
    (text, cur_actions, reward) = simulator.read()
    curr_node = simulator.controller.state
    curr_node_title = curr_node.title if parent_node_title is None else \
        f"{curr_node.title}_{simulator.controller.get_game_locals()}".replace(":", '=')
    graph.add_node(curr_node_title, state=curr_node, text=text, reward=reward)
    dot.node(curr_node_title, label=curr_node.title)
    if parent_node_title is not None:
        graph.add_edge(parent_node_title, curr_node_title, action=prev_path[-1])
        dot.edge(parent_node_title, curr_node_title, label=prev_path[-1])

    if curr_node_title in nodes_visited:
        return
    nodes_visited.add(curr_node_title)

    for choice in cur_actions:
        # move controller to the current state
        if curr_path != prev_path:
            simulator.restart()
            (_, actions, _) = simulator.read()
            curr_path.clear()
            for act in prev_path:
                curr_path.append(act)
                simulator.controller.transition(act)
                (_, actions, _) = simulator.read()

        (_, actions, _) = simulator.read()
        curr_path.append(choice)
        simulator.controller.transition(choice)

        dfs(simulator, curr_path.copy(), curr_node_title)


# def dfs(simulator, prev_path, parent_node_title):
#     (text, cur_actions, reward) = simulator.read()
#     curr_node = simulator.controller.state
#     curr_node_title =  curr_node.title if parent_node_title is None else \
#         f"{curr_node.title}_{simulator.controller.get_game_locals()}".replace(":", '=')
#     if curr_node.title not in graph.nodes:
#         graph.add_node(curr_node.title, text=text, reward=reward)
#         dot.node(curr_node.title)
#     if parent_node_title is not None and (parent_node_title, curr_node.title) not in graph.edges:
#         graph.add_edge(parent_node_title, curr_node.title, action=prev_path[-1])
#         dot.edge(parent_node_title, curr_node.title, label=prev_path[-1])
#
#     if curr_node_title in nodes_visited:
#         return
#     nodes_visited.add(curr_node_title)
#
#     for choice in cur_actions:
#         if curr_path != prev_path:
#             simulator.restart()
#             (_, actions, _) = simulator.read()
#             curr_path.clear()
#             for act in prev_path:
#                 curr_path.append(act)
#                 simulator.controller.transition(act)
#                 (_, actions, _) = simulator.read()
#
#         (_, actions, _) = simulator.read()
#         curr_path.append(choice)
#         simulator.controller.transition(choice)
#
#         dfs(simulator, curr_path.copy(), curr_node.title)
#
# dfs(simulator, [], None)

# shortest_paths = BiDict(nx.single_source_shortest_path_length(graph,'Final'))
#
# for k,l in shortest_paths.inverse.items():
#     c = graphviz.Digraph()
#     c.attr(rank='same')
#     for n in l:
#         c.node(n)
#     dot.subgraph(c)

# http://www.graphviz.org/docs/attrs/rank/

# os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
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


# folder_name = "html"
# Path(folder_name).mkdir(parents=True, exist_ok=True)
#
# for node in graph.nodes:
#     title = graph.nodes['Start']['state'].title
#     text = graph.nodes[node]['text']
#
#     # create page
#     d = dominate.document(title=title)
#     d += div(text, id='text', style="white-space: pre-line")
#     with d.add(div(id='choices')).add(ul()):
#         # create links
#         for next, data_next in graph.adj[node].items():
#             li(a(data_next['action'], href=f"{next}.html"))
#
#     with open(f"{folder_name}/{node if node != 'Start' else 'AAAA'}.html", "w") as file:
#         file.write(d.render())



# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(simulator.data, f, ensure_ascii=True, indent=4)



# folder_name = "html"
# Path(folder_name).mkdir(parents=True, exist_ok=True)
#
# d = dominate.document()
# with d.head:
#     meta(charset="utf-8")
#     script(type='text/javascript', src='js/bondage.min.js')
#     script(type='text/javascript', src='js/run.js')
#
# with d.body:
#     textarea(json.dumps(simulator.data, ensure_ascii=True), id='yarn', style='display:none')
#     div(id='display-area')
#
# with open("run.html", "w") as file:
#     file.write(d.render())