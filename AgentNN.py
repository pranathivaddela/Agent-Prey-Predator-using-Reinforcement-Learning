import random
import time
from utils import train_test_split
from Game import Game
from utils import experiment, create_graph
from simulators import complete_info_simulator
import pickle
import numpy as np
from NeuralNetwork import NeuralNetwork

# The U Agent implemented using the Neural Networks
class AgentNN:
    def __init__(self, game: Game, model_and_lookup):
        self.game = game
        self.utility_model, self.lookup_utilities = model_and_lookup

        self.game.register_agent(self)
        self.agent_loc = self.initialise_agent_location()
        self.edges_data = []
        for edge in self.game.graph.edges:
            self.edges_data += edge

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
        # print(self.agent_loc)

        return self.agent_loc

    def get_agent_next_location(self, prey_loc, predator_loc):
        agent_loc = self.get_location()
        agent_neighbors = self.game.get_neighbors(agent_loc) + [agent_loc]

        min_utility, next_loc = float('inf'), None


        for possible_next_loc in agent_neighbors:
            utility_val = None
            if (possible_next_loc, predator_loc, prey_loc) in self.lookup_utilities:
                utility_val = self.lookup_utilities[(possible_next_loc, predator_loc, prey_loc)]
            else:
                next_state = [possible_next_loc, predator_loc, prey_loc] + \
                             [self.game.shortest_dist(possible_next_loc, prey_loc), self.game.shortest_dist(possible_next_loc, predator_loc),
                              self.game.shortest_dist(possible_next_loc, prey_loc)] + self.edges_data

                next_state = np.array(next_state).reshape(1, -1)
                utility_val = self.utility_model.predict(next_state)

            if min_utility >= utility_val:
                next_loc = possible_next_loc
                min_utility = utility_val

        return next_loc


def read_data(x_file_path, y_file_path):
    X, Y = None, None

    with open(x_file_path, 'rb') as f:
        X = pickle.load(f)

    with open(y_file_path, 'rb') as f:
        Y = pickle.load(f)

    return X, Y


if __name__ == '__main__':
    start = time.time()
    X, Y = read_data('X_data.pkl', 'Y_data.pkl')
    from numpy import inf
    infinity_idx = np.where(Y == inf)[0]
    zero_utility_idx = np.where(Y == 0)[0]

    look_up_utilities = {}
    for state, utility in zip(X[infinity_idx], Y[infinity_idx]):
        look_up_utilities[tuple(state[:3])] = utility

    # for state, utility in zip(X[zero_utility_idx], Y[zero_utility_idx]):
    #     look_up_utilities[tuple(state[:3])] = utility

    trainable_idx = list(set(range(len(X))) - set(infinity_idx))
    # trainable_idx = list(set(range(len(X))) - set(list(infinity_idx) + list(zero_utility_idx)))

    X, Y = X[trainable_idx], Y[trainable_idx]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    print(f'Train data (X_train) size: {X_train.shape}')
    print(f'Train data (Y_train) size: {Y_train.shape}')
    print(f'Test data (X_test) size: {X_test.shape}')
    print(f'Test data (Y_test) size: {Y_test.shape}')


    #Defining the architecture of the Neural Network
    # 3 hidden layers
    model = NeuralNetwork(input_neurons=X.shape[-1],
                          hidden_layers_neurons=[2, 4, 4], # Number of neurons for the Hidden layers
                          hidden_layers_activations=['leaky_relu', 'leaky_relu', 'leaky_relu'], # Activation functions for the hidden layers
                          output_neurons=1,
                          init_method='he',
                          output_activation='linear')

    model.fit(X_train, Y_train,
              X_test, Y_test, loss_fun='rmse',
              algo='MiniBatch', mini_batch_size=256, epochs=30, lr=0.0001)

    agentNN_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        agentNN_results[key] = []

    game = Game()
    graph = create_graph('edges', game)

    for steps_allowed in range(50, 301, 50):
        # Passing Agent1 class and complete information simulator for experimentation
        result = experiment(AgentNN, game, (model, look_up_utilities), num_simulations=3000,
                            max_steps_allowed=steps_allowed, simulator=complete_info_simulator)

        for key, val in result.items():
            agentNN_results[key].append(val)

    print(agentNN_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('agentNN_results.pkl', 'wb') as f:
        pickle.dump(agentNN_results, f)

