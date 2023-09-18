import random
from Game import Game


class Predator:

    def __init__(self, game: Game):
        self.game = game
        self.game.register_predator(self)
        self.predator_loc = random.choice(self.game.get_nodes_list())

    def get_location(self):
        return self.predator_loc

    def predator_movement(self):
        agent_loc = self.game.get_agent_loc()

        neighbors = self.game.get_neighbors(self.predator_loc)

        neighbors_dist = {}

        for neighbor in neighbors:
            dist = self.game.shortest_dist(neighbor, agent_loc)

            if dist not in neighbors_dist.keys():
                neighbors_dist[dist] = [neighbor]
            else:
                neighbors_dist[dist].append(neighbor)

        min_dist = min(neighbors_dist.keys())

        self.predator_loc = random.choice(neighbors_dist[min_dist])

    def distracted_predator_movement(self):
        moves_towards_agent = [False]*40 + [True]*60
        random.shuffle(moves_towards_agent)
        is_moving_towards_agent = random.choice(moves_towards_agent)

        if is_moving_towards_agent:
            self.predator_movement()
        else:
            neighbors = self.game.get_neighbors(self.predator_loc)
            self.predator_loc = random.choice(neighbors)