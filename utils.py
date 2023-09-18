import time
from tqdm import tqdm
from Game import Game
from Predator import Predator
from Prey import Prey
from Environment import Graph
import pickle
import os
from statistics import mean
import numpy as np

def integerity_check(probs):
    pass


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
        result, steps, hang, n_knows_prey, n_knows_predator\
                = simulator(game, agent, predator, prey, max_steps_allowed)
        # steps_dist[steps] += 1
        results.append(result)
        hangs.append(hang)

        exactly_knows_prey += n_knows_prey
        exactly_knows_predator += n_knows_predator
        total_steps += steps

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

    return simulation_result


def create_graph(edges_file_path, game):
    graph = None

    edges = None
    if os.path.exists(edges_file_path):
        print('loading existing graph..')
        with open(edges_file_path, 'rb') as f:
            edges = pickle.load(f)
        graph = Graph(50, game, edges)

        
    else:
        print('Generating new graph and saving it..')
        graph = Graph(50, game)
        with open(edges_file_path, 'wb') as f:
            pickle.dump(graph.edges, f)

    N = graph.get_num_nodes()
    # for i in range(N):
    #     print(i, graph.adjacency_list[i])

    return graph


def initialise_utilities(graph):
    # state is represented using a tuple (agent_loc, predator_loc, prey_loc)
    print('Initializing utilities..')
    utilities = {}

    num_nodes = graph.num_nodes
    states_initialised_till = 0
    initialised = set()

    three_same = set()
    # states where all three sharing the same node
    for location in range(num_nodes):
        state = (location, location, location)
        utilities[state] = float('inf')
        initialised.add(state)
        three_same.add(state)
        states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')

    # states where, both prey and predator residing in the same neighbor

    for agent_loc in range(num_nodes):
        cur_neighbors = graph.get_neighbors(agent_loc)

        for neighbor in cur_neighbors:
            predator_loc = prey_loc = neighbor
            state = (agent_loc, predator_loc, prey_loc)
            utilities[state] = float('inf')

            initialised.add(state)
            states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')

    # Agent and prey sharing the same node
    for location in range(num_nodes):
        agent_loc = prey_loc = location
        for predator_loc in range(num_nodes):
            if predator_loc == location:
                continue
            state = (agent_loc, predator_loc, prey_loc)
            utilities[state] = 0
            initialised.add(state)
            states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')

    # Agent and predator sharing same node
    for location in range(num_nodes):
        agent_loc = predator_loc = location
        for prey_loc in range(num_nodes):
            if prey_loc == location:
                continue
            state = (agent_loc, predator_loc, prey_loc)
            utilities[state] = float('inf')
            initialised.add(state)
            states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')

    # Agent's neighbour having only prey but not predator
    for agent_loc in range(num_nodes):
        cur_neighbors = graph.get_neighbors(agent_loc)
        possible_predator_locs = list(set(range(50)) - set(cur_neighbors))
        for neighbor in cur_neighbors:
            prey_loc = neighbor
            for predator_loc in possible_predator_locs:
                if predator_loc == agent_loc:
                    continue
                state = (agent_loc, predator_loc, prey_loc)
                utilities[state] = 1
                initialised.add(state)
                states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')

    # Agent's neighbor having only predator but not prey
    for agent_loc in range(num_nodes):
        cur_neighbors = graph.get_neighbors(agent_loc)
        possible_prey_locs = list(set(range(50)) - set(cur_neighbors))
        for neighbor in cur_neighbors:
            predator_loc = neighbor
            for prey_loc in possible_prey_locs:
                if prey_loc == agent_loc:
                    continue
                state = (agent_loc, predator_loc, prey_loc)
                utilities[state] = float('inf')
                initialised.add(state)
                states_initialised_till += 1

    print(f'{states_initialised_till} states are initialised as of now')


    print(f'{states_initialised_till} states are initialised as of now')
    print(len(initialised))

    learnable = set()
    for agent_loc in range(num_nodes):
        for predator_loc in range(num_nodes):
            for prey_loc in range(num_nodes):
                state = (agent_loc, predator_loc, prey_loc)
                if state not in utilities:
                    utilities[state] = graph.shortest_dist(agent_loc, prey_loc)
                    learnable.add(state)

    print(len(initialised) + len(learnable))

    return utilities, learnable


def get_agent_next_loc(cur_state, graph, utilities):
    ''' Moves to the possible state which has max utility '''
    agent_loc, predator_loc, prey_loc = cur_state
    agent_neighbors = graph.get_neighbors(agent_loc) + [agent_loc]

    min_utility, next_loc = float('inf'), None

    for possible_next_loc in agent_neighbors:
        next_state = (possible_next_loc, predator_loc, prey_loc)
        if min_utility >= utilities[next_state]:
            next_loc = next_state[0]
            min_utility = utilities[next_state]

    return next_loc


def shortest_options(agent_loc, predator_loc, graph):
    predator_neighbors = graph.get_neighbors(predator_loc)
    temp = {}

    for neighbor in predator_neighbors:
        dist = graph.shortest_dist(agent_loc, neighbor)
        if dist not in temp:
            temp[dist] = [neighbor]
        else:
            temp[dist].append(neighbor)

    min_dist = min(temp.keys())

    return temp[min_dist]


def get_tp_for_predator(agent_loc, predator_loc, graph, dp):
    if (agent_loc, predator_loc) in dp:
        return dp[(agent_loc, predator_loc)]

    best_options = shortest_options(agent_loc, predator_loc, graph)
    degree = graph.degree(predator_loc)
    neighbors = graph.get_neighbors(predator_loc)

    transition_probs = {neighbor: .4 * (1/degree) for neighbor in neighbors}

    for best_neighbor in best_options:
        transition_probs[best_neighbor] += (.6 * 1/(len(best_options)))

    dp[(agent_loc, predator_loc)] = transition_probs

    return transition_probs


# 
def converge_utilities_v3(utilities, learnable_states, graph, beta, acceptable_error):
    print('Converging utilities..')
    start = time.time()
    dp = {}

    error = float('inf')
    iteration = 0
    while error >= acceptable_error:
        # states_done = 0
        new_utilities = {}
        for cur_state in tqdm(learnable_states):
            # neighbor_states = set()
            agent_loc, predator_loc, prey_loc = cur_state

            next_agent_loc = get_agent_next_loc(cur_state=cur_state, graph=graph, utilities=utilities)

            agent_neighbors = graph.get_neighbors(agent_loc)
            predator_neighbors = graph.get_neighbors(predator_loc)
            prey_neighbors = graph.get_neighbors(prey_loc) + [prey_loc]


            agent_transition_prob = 1
            prey_transition_prob = 1 / (len(prey_neighbors))
            predator_transition_probs = get_tp_for_predator(agent_loc, predator_loc, graph, dp)

            min_utility = float('inf')
            cost = 1


            for agent_neigh in agent_neighbors:
                sum_prey_pred = 0
                for prey_neigh in prey_neighbors:
                    for predator_neigh in predator_neighbors:
                        sum_prey_pred += prey_transition_prob * predator_transition_probs[predator_neigh] * (
                        utilities[(agent_neigh, predator_neigh, prey_neigh)])
                U_neigh = cost + beta * sum_prey_pred
                min_utility = min(min_utility, U_neigh)
                agent_next_move = agent_neigh

            new_utilities[cur_state] = min_utility

            # states_done += 1
            # print(states_done)

        error = float('-inf')

        for state in learnable_states:
            error = max(error, abs(utilities[state] - new_utilities[state]))
            utilities[state] = new_utilities[state]

        iteration += 1
        # print(len(learnable_states))
        print(f'error is {round(error, 4)}')

    print("Total iterations:", iteration)
    end = time.time()
    print(f'time took for convergence is {end - start} secs')

    return utilities


def train_test_split(X, Y, test_size=0.2):
    indexes = np.arange(0, len(X))
    np.random.shuffle(indexes)
    train_split = 1-test_size
    train_idx = int(train_split * len(X))

    X_train, X_test = X[:train_idx], X[train_idx:]
    Y_train, Y_test = Y[:train_idx], Y[train_idx:]

    return X_train, X_test, Y_train, Y_test

def get_max_utility_states(utilities):
    max_utility = float('-inf')
    max_utility_states = []

    for state, utility in utilities.items():
        if utility != float('inf'):
            if utility > max_utility:
                max_utility = utility
                max_utility_states = [state]
            elif utility == max_utility:
                max_utility_states.append(state)

    print("Max Utility", max_utility)
    print("Max Utility State", max_utility_states)
    
    return max_utility, max_utility_states


