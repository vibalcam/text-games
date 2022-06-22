import pandas as pd
from tqdm.auto import trange

from typing import Callable, Optional

from helper.helper import save_pickle
from game.agent import Agent
from game.simulator import GraphSimulator, Simulator


def run(get_agent: Callable[..., Agent], 
    simulator: Simulator, 
    n_runs=100, 
    reward_key:str = 'r', 
    question_id_key:str = 'q',
    ending_key:str = 'final', 
    filename:Optional[str] = None,
    **kwargs
):
    """
    Runs the agent on the simulation for a certain number of runs and obtains the endings and decisions taken

    :param get_agent: function that returns the agent
    :param Simulator simulator: simulator for the game
    :param int n_runs: number of runs, defaults to 100
    :param str reward_key: extras attribute key for the reward, defaults to 'r'
    :param str question_id_key: extras attribute key for the question key, defaults to 'q'
    :param str ending_key: node attribute key for the type of ending, defaults to 'final'
    :param filename: if not None, the filename of the file where to save the results
    :param kwargs: other parameters for the `get_agent` function
    """
    
    endings = []
    decisions = []

    for k in trange(int(n_runs)):
        # reset simulator and agent
        simulator.restart()
        agent = get_agent(**kwargs)

        final = None
        dec_count = 0
        # run simulator
        state, choices, extras = simulator.read()
        while not simulator.is_finished():
            # get ending
            if (tmp := simulator.get_current_node_attr().get(ending_key, None)) is not None:
                final = (k, simulator.get_current_node_attr()[GraphSimulator.ATTR_TITLE], tmp)

            choice = agent.act(state=state, actions=choices)
            choice_num = choice if isinstance(choice, int) else choices.index(choice)
            state, choices, extras = simulator.transition(choice)

            # get decisions taken
            if (label := extras.get(reward_key, None)):
                label = float(label)
            decisions.append((k, dec_count, float(extras.get(question_id_key, -1)), choice_num, label))
            dec_count += 1


        if final is not None:
            endings.append(final)

    df_endings = pd.DataFrame(data=endings, columns=['run', 'title', 'kind'])
    df_decisions = pd.DataFrame(data=decisions, columns=['run', 'num', 'qid', 'choice', 'label'])
    res = dict(
        endings=df_endings, 
        decisions=df_decisions,
    )

    if filename is not None:
        save_pickle(res, filename)
    return res
