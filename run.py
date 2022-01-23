from simulator import YarnSimulator
from agent import RandomAgent

simulator = YarnSimulator()
agent = RandomAgent(15)

state, choices, reward = simulator.read()
while not simulator.is_finished():
    choice = agent.act(state, choices)
    print(f"Option chosen: {choice}")
    state, choices, reward = simulator.act(choice)
    print((state, choices, reward))
