import pandas as pd
from tqdm.auto import trange

from game.agent import Agent
from game.simulator import GraphSimulator, Simulator


def run(get_agent, simulator: Simulator, n_runs=100, reward_key:str = 'r', ending_key:str = 'final'):
    """
    Runs the agent on the simulation for a certain number of runs and obtains the endings and decisions taken

    :param get_agent: function that returns the agent
    :param Simulator simulator: simulator for the game
    :param int n_runs: number of runs, defaults to 100
    :param str reward_key: extras attribute key for the reward, defaults to 'r'
    :param str ending_key: node attribute key for the type of ending, defaults to 'final'
    """
    
# :param reward_key: attribute key for the reward
    endings = []
    decisions = []

    for k in trange(int(n_runs)):
        # reset simulator and agent
        simulator.restart()
        agent = get_agent()

        final = None
        # run simulator
        state, choices, extras = simulator.read()
        while not simulator.is_finished():
            # get ending
            if (tmp := simulator.get_current_node_attr().get(ending_key, None)) is not None:
                final = (simulator.get_current_node_attr()[GraphSimulator.ATTR_TITLE], tmp)

            choice = agent.act(state=state, actions=choices)
            state, choices, extras = simulator.transition(choice)

            # get extras from decision taken
            if (dec := extras.get(reward_key, None)) is not None:
                decisions.append(float(dec))

        endings.append(final)

    df_endings = pd.DataFrame(data=endings, columns=['title', 'kind'])
    df_decisions = pd.DataFrame(data=decisions, columns=['decision'])
    return dict(
        endings=df_endings, 
        decisions=df_decisions,
    )
