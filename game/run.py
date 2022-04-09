import pandas as pd
from tqdm.auto import trange

from game.agent import Agent
from game.simulator import GraphSimulator, Simulator


def run(get_agent, simulator: Simulator, n_runs=100):
    endings = []

    for k in trange(int(n_runs)):
        # reset simulator and agent
        simulator.restart()
        agent = get_agent()

        final = None
        # run simulator
        state, choices, extras = simulator.read()
        while not simulator.is_finished():
            if (tmp := simulator.get_current_node_attr().get('final', None)) is not None:
                final = (simulator.get_current_node_attr()[GraphSimulator.ATTR_TITLE], tmp)

            choice = agent.act(state=state, actions=choices)
            state, choices, extras = simulator.transition(choice)

        endings.append(final)

    df = pd.DataFrame(data=endings, columns=['title', 'kind'])
    return df
