class Game:

    def __init__(self):
        self.graph = None
        self.agent = None
        self.prey = None
        self.predator = None

    def register_agent(self, agent):
        self.agent = agent

    def register_prey(self, prey):
        self.prey = prey

    def register_predator(self, predator):
        self.predator = predator

    def register_graph(self, graph):
        self.graph = graph

    def get_agent_loc(self):
        return self.agent.get_location()

    def get_prey_loc(self):
        return self.prey.get_location()

    def get_predator_loc(self):
        return self.predator.get_location()

    def get_neighbors(self, node):
        return self.graph.get_neighbors(node)

    def shortest_dist(self, n1, n2):
        return self.graph.shortest_dist(n1, n2)

    def get_nodes_list(self):
        return self.graph.get_node_list()

    def get_num_nodes(self):
        return self.graph.get_num_nodes()

    def get_degree(self, node):
        return self.graph.degree(node)