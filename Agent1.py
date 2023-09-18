import random
import time
from Game import Game
from utils import experiment
from simulators import complete_info_simulator
import pickle


class Agent1:

    def __init__(self, game: Game):
        self.game = game
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

        self.agent_loc = self.select_next_location_odd_agents(prey_loc, predator_loc)

        return self.agent_loc

    def select_next_location_odd_agents(self, prey_loc, predator_loc):
        ''' For given prey and predator location, according to rules of Agent1
        mentioned in project, this function will select a node to move the agent to.'''

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

        # Condition 1 Neighbors that are closer to the Prey and farther to the Predator
        for i in range(len(neighbors)):
            if prey_dist[i] < agent_to_prey and predator_dist[i] > agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # Condition 2 Neighbors that are closer to the Prey and not closer to the Predator
        for i in range(len(neighbors)):
            if prey_dist[i] < agent_to_prey and predator_dist[i] == agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # Condition 3 - Neighbors that are not farther from the Prey and farther from the Predator.

        for i in range(len(neighbors)):
            if prey_dist[i] == agent_to_prey and predator_dist[i] > agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # Condition 4 - Neighbors that are not farther from the Prey and not closer to the Predator.

        for i in range(len(neighbors)):
            if prey_dist[i] == agent_to_prey and predator_dist[i] == agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # Condition 5 - Neighbors that are farther from the Predator

        for i in range(len(neighbors)):
            if predator_dist[i] > agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # Condition 6 - Neighbors that are not closer to the Predator

        for i in range(len(neighbors)):
            if predator_dist[i] == agent_to_predator:
                possible_agent_next_loc.append(neighbors[i])

        if len(possible_agent_next_loc) != 0:
            return random.choice(possible_agent_next_loc)

        # stay still
        return self.agent_loc


if __name__ == "__main__":
    agent1_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        agent1_results[key] = []

    start = time.time()
    for steps_allowed in range(50, 301, 50):
        # Passing Agent1 class and complete information simulator for experimentation
        result = experiment(Agent1, num_graphs=30, simulation_per_graph=100,
                            max_steps_allowed=steps_allowed, simulator=complete_info_simulator)

        for key, val in result.items():
            agent1_results[key].append(val)

    print(agent1_results)
    end = time.time()
    print(f'execution time {end-start} secs')

    with open('agent1_results.pkl', 'wb') as f:
        pickle.dump(agent1_results, f)