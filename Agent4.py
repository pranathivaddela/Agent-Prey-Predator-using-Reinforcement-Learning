import time
import numpy as np
import random
from Game import Game
from Agent2 import Agent2
from Agent3 import Agent3
from utilities import experiment
from simulators import partial_prey_info_simulator
import pickle



class Agent4(Agent2, Agent3):
    '''
        Inherits Agent2, and Agent3 that will make Agent4 to mimic Agent3 along with functionalities of Agent2
        Agent4 will be made
        to have methods and properties used in partial prey information setting from Agent3
        to have self.prey_beliefs from Agent3
        to have self.initialise_prey_beliefs() from Agent3
        to have self.update_prey_beliefs() from Agent3
        to have self.update_prey_beliefs_after_move() from Agent3
        to have self.survey_for_prey() from Agent3
        to have self.surely_knows_prey from Agent3

        to have methods to perform better than Agent3 using methods inherited from Agent2
        to have self.estimate_future_prey_loc() from Agent2
        to have self.select_next_location_even_agents() from Agent2
        will override self.agent_movement()
        '''
    def __init__(self, game: Game):
        Agent3.__init__(self, game)

    def agent_movement(self):
        predator_loc = self.game.get_predator_loc()
        prey_probs = np.array(self.prey_beliefs)
        prey_possible_locs = list(np.where(prey_probs == prey_probs.max())[0])

        prey_loc = random.choice(prey_possible_locs)
        future_prey_loc = self.estimate_future_prey_loc(prey_loc)

        self.agent_loc = self.select_next_location_even_agents(future_prey_loc, predator_loc)
        return self.agent_loc


if __name__ == "__main__":
    agent4_results = {}
    keys = ['Agent', 'no_of_simulations', 'max_steps_allowed', 'Success rate', 'Failure rate',
            'Hangs', '% knowing prey location', '% knowing predator location']
    for key in keys:
        agent4_results[key] = []

    idx = 0
    start = time.time()
    for steps_allowed in range(50, 301, 50):

        result = experiment(Agent4, num_graphs=30, simulation_per_graph=100,
                            max_steps_allowed=steps_allowed, simulator=partial_prey_info_simulator)

        for key, val in result.items():
            agent4_results[key].append(val)

        print(f'% of knowing exact prey location: {agent4_results["% knowing prey location"][idx]}')
        idx += 1

    print(agent4_results)
    end = time.time()
    print(f'execution time {end - start} secs')

    with open('agent4_results.pkl', 'wb') as f:
        pickle.dump(agent4_results, f)