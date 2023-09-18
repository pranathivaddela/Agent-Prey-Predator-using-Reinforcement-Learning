import random
import time
from copy import copy
from Game import Game
import numpy as np
from simulators import complete_info_simulator
from utilities import experiment
import pickle


class Agent2:

    def __init__(self, game: Game):
        self.game = game
        self.game.register_agent(self)
        self.agent_loc = self.initialise_agent_location()

    def get_location(self):
        ''' returns the location of agent '''
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

    def prey_transition_model(self, cur_beliefs):
        ''' Returns the new beliefs according to prey movement

        new_belief[node] = SUMMATION( cur_belief[neighbor] * 1/(degree(neighbor)+1)  ) '''
        num_nodes = self.game.get_num_nodes()
        new_beliefs = [0.] * num_nodes

        for node in self.game.get_nodes_list():
            # new_belief will be contributed from neighbors and node itself (prey can stay still)
            influenced_by = self.game.get_neighbors(node) + [node]
            edge_probs = [1 / (self.game.get_degree(n) + 1) for n in influenced_by]

            for i in range(len(influenced_by)):
                n = influenced_by[i]
                new_beliefs[node] += (cur_beliefs[n] * edge_probs[i])

        return copy(new_beliefs)

    def estimate_future_prey_loc(self, prey_loc):
        ''' given the prey location, this function will estimate the future location
            based on beliefs calculated using prey transition model '''
        agent_to_prey = self.game.shortest_dist(self.get_location(), prey_loc)

        future_prey_beliefs = np.zeros(self.game.get_num_nodes())
        future_prey_beliefs[prey_loc] = 1.

        # this variable stores 'after how many time stamps from now we want to estimate location of prey
        future_simulations = min(5, agent_to_prey // 2)
        for i in range(future_simulations):
            future_prey_beliefs = self.prey_transition_model(future_prey_beliefs)

        # selecting future prey location which having highest belief,
        # breaking ties using shortest distance to agent
        future_prey_beliefs = np.array(future_prey_beliefs)
        future_prey_locs = list(np.where(future_prey_beliefs == future_prey_beliefs.max())[0])

        future_prey_location = None
        if len(future_prey_locs) > 1:
            distances = [self.game.shortest_dist(self.get_location(), prey)
                         for prey in future_prey_locs]
            min_distance = min(distances)
            future_prey_options = []
            for node, dist in zip(future_prey_locs, distances):
                if dist == min_distance:
                    future_prey_options.append(node)

            future_prey_location = random.choice(future_prey_options)
        else:
            future_prey_location = future_prey_locs[0]

        return future_prey_location

    def agent_movement(self):
        ''' In even numbered agents, agent will make a move towards future prey location'''
        predator_loc = self.game.get_predator_loc()
        prey_loc = self.game.get_prey_loc()
        future_prey_loc = self.estimate_future_prey_loc(prey_loc)

        self.agent_loc = self.select_next_location_even_agents(future_prey_loc, predator_loc)
        return self.agent_loc

    def select_next_location_even_agents(self, prey_loc, predator_loc):
        agent_loc = self.get_location()
        neighbors = self.game.get_neighbors(agent_loc)

        prey_dist = []
        predator_dist = []

        for neighbor in neighbors:
            prey_dist.append(self.game.shortest_dist(neighbor, prey_loc))
            predator_dist.append(self.game.shortest_dist(neighbor, predator_loc))

        agent_to_prey = self.game.shortest_dist(agent_loc, prey_loc)
        agent_to_predator = self.game.shortest_dist(agent_loc, predator_loc)

        possible_agent_next_loc = []

        for i in range(len(neighbors)):
            if prey_dist[i] < agent_to_prey and predator_dist[i] > agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        for i in range(len(neighbors)):
            if prey_dist[i] < agent_to_prey and predator_dist[i] == agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        for i in range(len(neighbors)):
            if predator_dist[i] > agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        return self.agent_loc


if __name__ == "__main__":
    agent2_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']

    for key in keys:
        agent2_results[key] = []

    start = time.time()
    for steps_allowed in range(50, 301, 50):
        # Passing Agent2 class and complete information simulator for experimentation
        result = experiment(Agent2, num_graphs=30, simulation_per_graph=100,
                            max_steps_allowed=steps_allowed, simulator=complete_info_simulator)

        for key, val in result.items():
            agent2_results[key].append(val)

    print(agent2_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('agent2_results.pkl', 'wb') as f:
        pickle.dump(agent2_results, f)