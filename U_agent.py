import random
import time
from Game import Game
from utils import *
from simulators import complete_info_simulator
import pickle
import numpy as np


class U_agent:

    def __init__(self, game: Game, utilities):
        self.game = game
        self.utilities = utilities
        self.game.register_agent(self)
        self.agent_loc = self.initialise_agent_location()

    def get_location(self):
        return self.agent_loc

    def initialise_agent_location(self):
        ''' selects a location for agent to spawn from
         unoccupied nodes in graph. '''
        prey_loc = self.game.get_prey_loc()
        predator_loc = self.game.get_predator_loc()
        unavailable_locs = set([prey_loc, predator_loc])

        node_list = self.game.get_nodes_list()
        for loc in unavailable_locs:
            node_list.remove(loc)

        agent_loc = random.choice(node_list)

        return agent_loc

    def agent_movement(self):
        prey_loc = self.game.get_prey_loc()
        predator_loc = self.game.get_predator_loc()

        self.agent_loc = self.get_agent_next_location(prey_loc, predator_loc)

        return self.agent_loc

    def get_agent_next_location(self, prey_loc, predator_loc):
        agent_loc = self.get_location()
        agent_neighbors = self.game.get_neighbors(agent_loc) + [agent_loc]

        min_utility, next_loc = float('inf'), None

        for possible_next_loc in agent_neighbors:
            next_state = (possible_next_loc, predator_loc, prey_loc)
            if min_utility >= self.utilities[next_state]:
                next_loc = next_state[0]
                min_utility = self.utilities[next_state]

        return next_loc

# To create the dataset, the datapoints from the Agent simulations
def create_dataset(utilities, graph):
    X = list()
    Y = list()
    edges = graph.edges

    edges_data = []
    for edge in edges:
        edges_data += edge

    for state in utilities:


        agent_loc, predator_loc, prey_loc = state
        x_i = list(state) + \
              [graph.shortest_dist(agent_loc, prey_loc), graph.shortest_dist(agent_loc, predator_loc),
               graph.shortest_dist(predator_loc, prey_loc)] + edges_data
        y_i = utilities[state]
        X.append(x_i)
        Y.append(y_i)

    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    with open('X_data.pkl', 'wb') as f:
        pickle.dump(X, f)

    with open('Y_data.pkl', 'wb') as f:
        pickle.dump(Y, f)


if __name__ == "__main__":
    start = time.time()
    Uagent_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        Uagent_results[key] = []

    game = Game()
    graph = create_graph('edges', game)
    utilities, learnable_states = initialise_utilities(graph)
    utilities = converge_utilities_v3(utilities, learnable_states, graph, beta=0.7, acceptable_error=0.001)

    get_max_utility_states(utilities)
    # utilities_values = set()
    # for key, value in utilities.items():
    #     # if value == 0:
    #     #     print(value)
    #     utilities_values.add(round(value, 3))

    # print(f'different utility values count: {len(utilities_values)}')
    # print(sorted(list(utilities_values)))


    # Creating the Dataset to feed the Neural Networks as input
    create_dataset(utilities, graph)

    for steps_allowed in range(50, 301, 50):
        # Passing Agent1 class and complete information simulator for experimentation
        result = experiment(U_agent, game, utilities, num_simulations=3000,
                            max_steps_allowed=steps_allowed, simulator=complete_info_simulator)

        for key, val in result.items():
            Uagent_results[key].append(val)

    print(Uagent_results)
    end = time.time()
    print(f'execution time {end-start} secs')

    with open('Uagent_results.pkl', 'wb') as f:
        pickle.dump(Uagent_results, f)