from utils import *
from Game import Game
from Prey import Prey
from Predator import Predator
from U_agent import U_agent

if __name__ == '__main__':
    print('hello')
    game = Game()
    graph = create_graph('edges', game)
    utilities, learnable_states = initialise_utilities(graph)

    prey = Prey(game)
    predator = Predator(game)
    agent = U_agent(game, utilities)
    utilities = converge_utilities(utilities, learnable_states, graph, beta=0.1, acceptable_error=0.001)


