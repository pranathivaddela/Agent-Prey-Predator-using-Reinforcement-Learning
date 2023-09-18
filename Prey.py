import random
from Environment import Graph
from Game import Game


class Prey:

    def __init__(self, game: Game):
        self.game = game
        self.game.register_prey(self)
        self.prey_loc = random.choice(self.game.get_nodes_list())

        # print("Prey init loc:", self.prey_loc)

    def get_location(self):
        return self.prey_loc

    def prey_movement(self):
        neighbors = self.game.get_neighbors(self.prey_loc)

        neighbors.append(self.prey_loc)

        self.prey_loc = random.choice(neighbors)

        # print("Prey next loc:", self.prey_loc)


if __name__ == "__main__":
    g = Graph(30)
    prey = Prey(g)

    # prey.prey_movement()