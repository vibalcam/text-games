from simulator import YarnSimulator

simulator = YarnSimulator()

while not simulator.is_finished():
    print(simulator.read())
    print(simulator.act(0))
