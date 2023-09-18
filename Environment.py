import os.path
import random
from copy import copy
import pickle


class Graph:

    def __init__(self, num_nodes, game, edges=None):

        self.game = game
        self.game.register_graph(self)
        self.adjacency_list = {}
        self.num_nodes = num_nodes
        self.nodes_list = list(range(self.num_nodes))
        self.edges = []
        self.re = 0

        if edges is None:
            self.create_graph()
        else:
            self.create_graph_from(edges)
        self.calculate_shortest_distances()
        # print('####################################################################')
        # print(max(self.distance))

    def create_graph_from(self, edges):
        for edge in edges:
            n1, n2 = edge
            self.add_edge(n1, n2)

    def get_node_list(self):
        return copy(self.nodes_list)

    def get_num_nodes(self):
        return self.num_nodes

    def create_graph(self):
        
        self.create_cycle()
        # print("Cycle:",self.adjacency_list)
        node_list = self.get_node_list()
        random.shuffle(node_list)
        # print("Set:",self.nodes_set)
        
        for i in node_list:
            if self.degree(i) == 2:
                self.random_edge(i)

        # for i in range(self.num_nodes):
        #     print(i, self.adjacency_list[i])
        # print(self.re)

    def create_cycle(self):
        for i in range(self.num_nodes):
            if i == self.num_nodes - 1:
                self.add_edge(i, 0)
            else:
                self.add_edge(i, i+1)
                # print(i, i+1)

    def add_edge(self, node1, node2):
        if node1 not in self.adjacency_list.keys():
            self.adjacency_list[node1] = []
            
        if node2 not in self.adjacency_list.keys():
            self.adjacency_list[node2] = []  

        self.adjacency_list[node1].append(node2)
        self.adjacency_list[node2].append(node1)
        self.edges.append([node1, node2])

    def degree(self, node):
        # print("neighbours of the node:", node, self.adjacency_list[node])
        return len(self.adjacency_list[node])
    
    def random_edge(self, node):

        available_nodes = list(range((node-5), (node + 6)))
        available_nodes.remove(node)
        available_nodes.remove(node-1)
        available_nodes.remove(node+1)

        # print("available:",available_nodes)

        an = [i%self.num_nodes for i in available_nodes]
        
        # for i in range(len(available_nodes)):
        #     an.append(available_nodes[i]%self.num_nodes)

        while len(an) > 0:
            next_node = random.choice(an)

            if self.degree(next_node) == 2:
                self.add_edge(node, next_node)
                # print(self.re+1," randomly added",node, next_node) 
                self.re += 1
                break
            else:
                an.remove(next_node)

    def get_neighbors(self, node):
        neighbours = self.adjacency_list[node]
        return copy(neighbours)

    def calculate_shortest_distances(self):
        infinity = float('inf')
        distance = [[infinity for col in range(self.num_nodes)] for row in range(self.num_nodes)]

        for edge in self.edges:
            x, y = edge
            distance[x][y] = 1
            distance[y][x] = 1

        for node in self.nodes_list:
            distance[node][node] = 0

        for k in self.nodes_list:
            for i in self.nodes_list:
                for j in self.nodes_list:
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

        self.distance = distance

    def shortest_dist(self, n1, n2):
        return self.distance[n1][n2]

    # def shortest_dist(self, n1, n2):
    #     # print(n1, n2)
    #     queue = []
    #     queue.append(n1)
    #
    #     visited = [False]*self.num_nodes
    #     visited[n1] = True
    #
    #     dist = {}
    #     dist[n1] = 0
    #
    #     while queue:
    #         x = queue.pop(0)
    #         if x == n2:
    #             return dist[x]
    #         else:
    #             for i in self.adjacency_list[x]:
    #                 if not visited[i]:
    #                     dist[i] = dist[x] + 1
    #                     queue.append(i)
    #                     visited[i] = True


if __name__ == "__main__":
    
    graph = None