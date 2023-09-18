import os.path
import pickle
import random
import numpy as np
import time
from Game import Game
from copy import copy
from U_agent import U_agent
from Predator import Predator
from Prey import Prey
from simulators import partial_prey_info_simulator
from utils import initialise_utilities, create_graph, converge_utilities_v3


# from utils import *

# The Upartial Agent implemented in the Distracted Predator Environment
class Upartial_agent(U_agent):
    '''
    Inherits U_agent, that will make Upartial_agent
    to have self.agent_loc from U_agent
    to have self.game from U_agent
    to have self.initialise_agent_location() from U_agent
    to have self.select_next_location_odd_agents() from U_agent
    will override self.agent_movement() from U_agent

    and define new methods required for partial prey information setting
    '''
    def __init__(self, game: Game, utilities):
        super().__init__(game, utilities)
        self.prey_beliefs = None
        self.initialise_prey_beliefs()
        self.data_X = []
        self.data_Y = []
        self.total_edges = []

        for edge in game.graph.edges:
            self.total_edges += edge
        # print(self.agent_loc)

    def initialise_prey_beliefs(self):
        self.prey_beliefs = [1. / 49 for node in (self.game.get_nodes_list())]
        self.prey_beliefs[self.agent_loc] = 0

    def update_prey_beliefs_after_move(self):
        num_nodes = self.game.get_num_nodes()
        probs = [0.] * num_nodes

        for node in self.game.get_nodes_list():
            influenced_by = self.game.get_neighbors(node) + [node]
            edge_probs = [1/(self.game.get_degree(n)+1) for n in influenced_by]

            for i in range(len(influenced_by)):
                n = influenced_by[i]
                probs[node] += (self.prey_beliefs[n] * edge_probs[i])

        self.prey_beliefs = copy(probs)
        return copy(probs)

    def update_prey_beliefs(self, knowledge_node, is_prey_present):
        if is_prey_present:
            self.prey_beliefs = [0.] * self.game.get_num_nodes()
            self.prey_beliefs[knowledge_node] = 1.
            return copy(self.prey_beliefs)

        pr_knowledge_node = self.prey_beliefs[knowledge_node]

        for node in self.game.get_nodes_list():
            if node == knowledge_node:
                self.prey_beliefs[node] = 0.
            else:
                self.prey_beliefs[node] = self.prey_beliefs[node] / (1-pr_knowledge_node)

        return copy(self.prey_beliefs)

    def agent_movement(self):
        predator_loc = self.game.get_predator_loc()

        self.agent_loc = self.get_agent_next_location(predator_loc)
        # print("Agent Location", self.agent_loc)
        return self.agent_loc

    def get_agent_next_location(self, predator_loc):
        agent_loc = self.agent_loc
        agent_possible_locations = []
        neighbour_utilities = []

        neighbours = self.game.get_neighbors(agent_loc) + [agent_loc]

        agent_next_loc = None
        minimum_utility = float('inf')

        prey_probs = np.array(self.prey_beliefs)
        prey_max_belief_loc = random.choice(np.where(prey_probs == prey_probs.max())[0])

        # Calculating the expected Utility for the transition state(neighbour)
        # Agent only moves to the neighbour which has minimum Expected Utility
        for neigh in neighbours:
            U_partial = 0
            for prey_loc in range(len(self.prey_beliefs)):
                if self.prey_beliefs[prey_loc] == 0:
                    continue
                U = self.utilities[(neigh, predator_loc, prey_loc)]
                U_partial += self.prey_beliefs[prey_loc]*U

            # Data point creation for Neural Network
            data_point_x = [neigh, predator_loc, prey_max_belief_loc] + \
                           [self.game.shortest_dist(neigh, prey_max_belief_loc), self.game.shortest_dist(neigh, predator_loc),
                            self.game.shortest_dist(predator_loc, prey_max_belief_loc)] + \
                           self.prey_beliefs + self.total_edges
            # print(len(data_point_x))
            data_point_y = [U_partial]

            self.data_X.append(data_point_x)
            self.data_Y.append(data_point_y)

            neighbour_utilities.append(U_partial)
        minimum_utility = min(neighbour_utilities)

        for neigh, utility in zip(neighbours, neighbour_utilities):
            # print("Utility", utility)
            if utility == minimum_utility:
                agent_possible_locations.append(neigh)

        # print(len(agent_possible_locations))
        return random.choice(agent_possible_locations)

    def survey_for_prey(self):
        probs = np.array(self.prey_beliefs)
        survey_options = np.where(probs == probs.max())[0]

        survey_options = list(survey_options)
        survey_node = random.choice(survey_options)
        is_prey_present = (survey_node == self.game.get_prey_loc())

        return survey_node, is_prey_present

    def surely_knows_prey(self):
        beliefs = self.prey_beliefs
        if max(beliefs) == 1:
            return True

        return False


def experiment(Agentclass, game, utilities, num_simulations, max_steps_allowed, simulator):
    exactly_knows_prey = 0
    exactly_knows_predator = 0
    total_steps = 0
    data_X = []
    data_Y = []

    results = []
    hangs = []

    for j in range(num_simulations):
        predator = Predator(game)
        prey = Prey(game)
        agent = Agentclass(game, utilities)
        result, steps, hang, n_knows_prey, n_knows_predator \
                = simulator(game, agent, predator, prey, max_steps_allowed)
        # steps_dist[steps] += 1
        results.append(result)
        hangs.append(hang)

        exactly_knows_prey += n_knows_prey
        exactly_knows_predator += n_knows_predator
        total_steps += steps

        if Agentclass == Upartial_agent:
            data_X += agent.data_X
            data_Y += agent.data_Y

    success_rate = sum(results) / len(results)
    hang_rate = sum(hangs) / len(hangs)
    failure_rate = 1 - (success_rate + hang_rate)

    overall_success_rate = round(success_rate, 4)
    overall_hang_rate = round(hang_rate, 4)
    overall_failure_rate = round(failure_rate, 4)

    rates = [overall_success_rate, overall_failure_rate, overall_hang_rate]

    print('--------------- Configuration for this experiment ---------------')
    print(f'Agent is of type: {type(agent).__name__}')
    print(f'Number of simulations: {num_simulations}')
    print(f'Maximum number of steps allowed in a game(steps threshold): {max_steps_allowed}')
    print('-----------------------------------------------------------------')

    print(f'Results of this experiment ({type(agent).__name__}) as follows...\n')
    print("{:<15} {:<15} {:<15}".format('Success rate', 'Failure rate', 'Hang rate'))
    print("{:<15} {:<15} {:<15}".format(*rates))
    print()

    perc_knowing_prey = round(100*(exactly_knows_prey / total_steps), 2)
    perc_knowing_predator = round(100*(exactly_knows_predator / total_steps), 2)

    simulation_result = {'Agent': type(agent).__name__,
                         'no_of_simulations': num_simulations,
                         'max_steps_allowed': max_steps_allowed,
                         'Success rate': overall_success_rate,
                         'Failure rate': overall_failure_rate,
                         'Hangs': sum(hangs),
                         '% knowing prey location': perc_knowing_prey,
                         '% knowing predator location': perc_knowing_predator}

    if Agentclass == Upartial_agent:
        X = np.array(data_X)
        Y = np.array(data_Y).reshape(-1, 1)

        if os.path.exists('X_partial_data.pkl') and os.path.exists('Y_partial_data.pkl'):
            existing_data_x = None
            with open('X_partial_data.pkl', 'rb') as f:
                existing_data_x = pickle.load(f)
            # print(X.shape, existing_data_x.shape)
            X = np.concatenate((existing_data_x, X), axis=0)

            existing_data_y = None
            with open('Y_partial_data.pkl', 'rb') as f:
                existing_data_y = pickle.load(f)
            Y = np.concatenate((existing_data_y, Y), axis=0)

        with open('X_partial_data.pkl', 'wb') as f:
            pickle.dump(X, f)

        with open('Y_partial_data.pkl', 'wb') as f:
            pickle.dump(Y, f)

    return simulation_result


if __name__ == "__main__":
    start = time.time()
    Up_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        Up_results[key] = []

    game = Game()
    graph = create_graph('edges', game)

    utilities, learnable_states = initialise_utilities(graph)
    utilities = converge_utilities_v3(utilities, learnable_states, graph, beta=0.7, acceptable_error=0.001)

    for steps_allowed in range(50, 301, 50):
        # Passing Agent1 class and complete information simulator for experimentation
        result = experiment(Upartial_agent, game, utilities, num_simulations=3000,
                            max_steps_allowed=steps_allowed, simulator=partial_prey_info_simulator)

        for key, val in result.items():
            Up_results[key].append(val)

    print(Up_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('Upartial_results.pkl', 'wb') as f:
        pickle.dump(Up_results, f)