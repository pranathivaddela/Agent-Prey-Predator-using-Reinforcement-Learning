import random
import time
from utils import train_test_split, initialise_utilities, converge_utilities_v3
from Game import Game
from Upartial_agent import Upartial_agent
from utils import experiment, create_graph
from simulators import complete_info_simulator
import pickle
import numpy as np
from NeuralNetwork import NeuralNetwork


class AgentPartialNN(Upartial_agent):
    def __init__(self, game: Game, model_and_lookup):
        self.game = game
        self.game.register_agent(self)
        self.agent_loc = self.initialise_agent_location()

        self.utility_model, self.lookup_utilities = model_and_lookup

        self.prey_beliefs = None
        self.initialise_prey_beliefs()

        self.edges_data = []
        for edge in self.game.graph.edges:
            self.edges_data += edge

        # print(len(self.edges_data))

    def agent_movement(self):
        predator_loc = self.game.get_predator_loc()
        prey_probs = np.array(self.prey_beliefs)
        prey_loc = random.choice(np.where(prey_probs == np.max(prey_probs))[0])

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
                              self.game.shortest_dist(predator_loc, prey_loc)] + \
                             list(np.array(self.prey_beliefs)*100) + self.edges_data

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
    X, Y = read_data('X_partial_data.pkl', 'Y_partial_data.pkl')
    X[:][6:55] *= 100
    from numpy import inf
    infinity_idx = np.where(Y == inf)[0]
    zero_utility_idx = np.where(Y == 0)[0]

    game = Game()
    graph = create_graph('edges', game)

    utilities, learnable_states = initialise_utilities(graph)
    utilities = converge_utilities_v3(utilities, learnable_states, graph, beta=0.7, acceptable_error=0.001)

    look_up_utilities = {}

    for state, utility in utilities.items():
        if utility == float('inf'):
            look_up_utilities[state] = utility

    # trainable_idx = list(set(range(len(X))) - set(infinity_idx))
    trainable_idx = list(set(range(len(X))) - set(list(infinity_idx) + list(zero_utility_idx)))

    X, Y = X[trainable_idx], Y[trainable_idx]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    print(f'Train data (X_train) size: {X_train.shape}')
    print(f'Train data (Y_train) size: {Y_train.shape}')
    print(f'Test data (X_test) size: {X_test.shape}')
    print(f'Test data (Y_test) size: {Y_test.shape}')

    model = NeuralNetwork(input_neurons=X.shape[-1],
                          hidden_layers_neurons=[36, 12, 4],
                          hidden_layers_activations=['tanh', 'leaky_relu', 'leaky_relu'],
                          output_neurons=1,
                          init_method='he',
                          output_activation='linear')

    model.fit(X_train, Y_train,
              X_test, Y_test, loss_fun='mse',
              algo='MiniBatch', mini_batch_size=1024, epochs=10, lr=0.001)

    agentPartialNN_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        agentPartialNN_results[key] = []

    for steps_allowed in range(50, 301, 50):
        # Passing Agent1 class and complete information simulator for experimentation
        result = experiment(AgentPartialNN, game, (model, look_up_utilities), num_simulations=3000,
                            max_steps_allowed=steps_allowed, simulator=complete_info_simulator)

        for key, val in result.items():
            agentPartialNN_results[key].append(val)

    print(agentPartialNN_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('agentNNpartial_results.pkl', 'wb') as f:
        pickle.dump(agentPartialNN_results, f)