import pickle
import random
import numpy as np
import time
from utilities import experiment
from Game import Game
from copy import copy
from Agent1 import Agent1
from simulators import partial_prey_info_simulator


class Agent3(Agent1):
    '''
    Inherits Agent1, that will make Agent3
    to have self.agent_loc from Agent1
    to have self.game from Agent1
    to have self.initialise_agent_location() from Agent1
    to have self.select_next_location_odd_agents() from Agent1
    will override self.agent_movement() from Agent1

    and define new methods required for partial prey information setting
    '''
    def __init__(self, game: Game):
        super().__init__(game)
        self.prey_beliefs = None
        self.initialise_prey_beliefs()

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
        prey_probs = np.array(self.prey_beliefs)
        prey_possible_locs = list(np.where(prey_probs == prey_probs.max())[0])

        prey_loc = random.choice(prey_possible_locs)
        predator_loc = self.game.get_predator_loc()

        self.agent_loc = self.select_next_location_odd_agents(prey_loc, predator_loc)

        return self.agent_loc

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


if __name__ == "__main__":
    agent3_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        agent3_results[key] = []

    idx = 0
    start = time.time()
    for steps_allowed in range(50, 301, 50):
        # Passing Agent3 class and partial prey information simulator for experimentation
        result = experiment(Agent3, num_graphs=30, simulation_per_graph=100,
                            max_steps_allowed=steps_allowed, simulator=partial_prey_info_simulator)

        for key, val in result.items():
            agent3_results[key].append(val)

        print(f'% of knowing exact prey location: {agent3_results["% knowing prey location"][idx]}')
        idx += 1

    print(agent3_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('agent3_results.pkl', 'wb') as f:
        pickle.dump(agent3_results, f)