from agent import RandomAgent
from simulator import YarnSimulator

n_runs = 1

simulator = YarnSimulator(text_unk_macro="")

for k in range(n_runs):
    # restart simulator
    simulator.restart()

    # create agent
    agent = RandomAgent(15)
    # agent = TorchModelAgent(
    #     model_path=Path('models/saved/adamw_max_val_acc_8_False_125,[20],[20]_0.001'),
    #     rand=0,
    #     use_cpu=False,
    #     seed=123,
    # )

    # run game
    state, choices, reward = simulator.read()
    while not simulator.is_finished():
        choice = agent.act(state, choices)
        # print(f"Option chosen: {choice}")
        state, choices, reward = simulator.transition(choice)
        # print((state, choices, reward))

    # get result
    print(simulator)
