from agent import RandomAgent
from simulator import YarnSimulator

simulator = YarnSimulator(text_unk_macro="")
agent = RandomAgent(15)

state, choices, reward = simulator.read()
while not simulator.is_finished():
    choice = agent.act(state, choices)
    print(f"Option chosen: {choice}")
    state, choices, reward = simulator.transition(choice)
    print((state, choices, reward))
