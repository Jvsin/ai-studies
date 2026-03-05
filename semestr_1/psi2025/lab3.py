import math
import random
from collections import deque
from itertools import permutations
import time
import heapq

import numpy as np


def random_cities(city_counter):
    cities = []
    for i in range(city_counter):
        new_city = (random.randint(-100, 100), random.randint(-100, 100),
                    random.randint(0, 50))
        print(f"{i}: {new_city}")
        cities.append(new_city)
    return cities


def euclidian_dist(a, b, symmetrical=True):
    result = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    if not symmetrical:
        if a[2] > b[2]:
            result = 0.9 * result
        elif a[2] < b[2]:
            result = 1.1 * result
    return result


def count_distances(cities, cut_connection=0.2, symmetrical=False):
    size = len(cities)
    dist_list = []
    for i in range(size):
        city = []
        for j in range(size):
            city.append(euclidian_dist(cities[i], cities[j], symmetrical))
        dist_list.append(city)

    num_to_cut = int((size ** 2 - size) * cut_connection)
    all_indices = [(i, j) for i in range(len(cities)) for j in range(len(cities)) if i != j]
    to_cut = random.sample(all_indices, num_to_cut)

    for i, j in to_cut:
        dist_list[i][j] = -1

    return dist_list


def print_distance_matrix(matrix):
    print("       " + "        ".join(map(str, range(len(matrix)))))
    for i, row in enumerate(matrix):
        row_values = "    ".join(f"{val:.2f}" for val in row)
        print(f"{i}: {row_values}")


def create_tree_graph(cities_matrix):
    graph = {}
    for i in range(len(cities_matrix)):
        city_routes = []
        for j in range(len(cities_matrix)):
            if cities_matrix[i][j] != -1 and i != j:
                city_routes.append(j)
        graph[i] = city_routes
    # print(f"\nCities graph: {graph}")
    return graph


def nearest_neigh(cities_matrix, start=0):
    size = len(cities_matrix)
    path = [start]
    cost = 0
    current_city = start

    tree = create_tree_graph(cities_matrix)
    while len(path) != size + 1:
        # print(current_city, path, cost)

        if len(path) == size:
            if start in tree[current_city]:
                # print(f"\nBack to start. Cost: {cities_matrix[current_city][start]}")
                path.append(start)
                cost += cities_matrix[current_city][start]
                return path, cost
            else:
                print("There's no possible path.")
                return None, 0
        next_moves = {}
        for next_city in tree[current_city]:
            if next_city not in path:
                next_moves[next_city] = cities_matrix[current_city][next_city]
        if len(next_moves):
            # print(f"\nPotential next moves: {next_moves}")
            min_key, min_cost = min(next_moves.items(), key=lambda x: x[1])
            path.append(min_key)
            cost += min_cost
            current_city = min_key
        else:
            print("There's no possible path.")
            return None, 0


def count_route_cost(route, matrix):
    route_cost = 0
    for i in range(len(matrix) - 1):
        if matrix[route[i]][route[i + 1]] == -1:
            return 0
        route_cost += matrix[route[i]][route[i + 1]]
    return route_cost



def bfs_salesman(cities_matrix, start=0):
    best_path = None
    lowest_cost = float('inf')

    tree = create_tree_graph(cities_matrix)
    queue = deque([(start, [start], 0)])
    while queue:
        curr_city, path, cost = queue.popleft()
        # print(curr_city, path, cost)

        if len(path) == len(cities_matrix):
            if start in tree[curr_city]:
                cost += cities_matrix[curr_city][start]
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path + [start]
                    # print(f"Found best way: {best_path} with cost: {cost}")
            continue

        for next_city in tree[curr_city]:
            if next_city not in path:
                queue.append((next_city, path + [next_city], cost + cities_matrix[curr_city][next_city]))

    return best_path, lowest_cost


def dfs_salesman(cities_matrix, start=0):
    best_path = None
    lowest_cost = float('inf')

    tree = create_tree_graph(cities_matrix)
    stack = [(start, [start], 0)]
    while stack:
        curr_city, path, cost = stack.pop()
        # print(curr_city, path, cost)

        if len(path) == len(cities_matrix):
            if start in tree[curr_city]:
                cost += cities_matrix[curr_city][start]
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path + [start]
                    # print(f"Found best way: {best_path} with cost: {cost}")
            continue

        for next_city in reversed(tree[curr_city]):
            if next_city not in path:
                stack.append((next_city, path + [next_city], cost + cities_matrix[curr_city][next_city]))

    return best_path, lowest_cost

def heuristic_admissible(matrix, city, unvisited):
    if not unvisited:
        return matrix[city][0] if matrix[city][0] != -1 else float('inf')
    # print(f"Jesteśmy w {city}, następne miasta to: {unvisited}")

    remain_steps = len(unvisited) + 1
    first_move = [matrix[city][next_city] for next_city in unvisited if matrix[city][next_city] != -1]
    if len(first_move) == 0:
        return float('inf')

    min_edge = min(first_move)
    for i in unvisited:
        for j in unvisited:
            if matrix[i][j] != -1 and i != j:
                min_edge = min(min_edge, matrix[i][j])
        if matrix[i][0] != -1 and i != 0:
            min_edge = min(min_edge, matrix[i][0])
    # print(min_edge)
    return min_edge * remain_steps if min_edge != float('inf') else float('inf')


def heuristic_inadmissible(matrix, city, unvisited):
    if not unvisited:
        return matrix[city][0] if matrix[city][0] != -1 else float('inf')

    remain_steps = len(unvisited) + 1
    roads = []
    for next_city in unvisited:
        if matrix[city][next_city] != -1:
            roads.append(matrix[city][next_city])
    for i in unvisited:
        for j in unvisited:
            if matrix[i][j] != -1 and i != j:
                roads.append(matrix[i][j])
        # if matrix[i][0] != -1 and i != 0:
        #     roads.append(matrix[i][0])
    # print(roads)
    result = np.mean(roads) if len(roads) != 0 else float('inf')
    # print(result)
    return result * remain_steps


def astar_tsp(matrix, heuristic_func, start=0):
    size = len(matrix)
    tree = create_tree_graph(matrix)
    heap = [(0 + heuristic_func(matrix, start, set(range(size)) - {start}), 0, start, [start])]
    lowest_cost = float('inf')
    best_path = None
    while heap:
        est_total_cost, curr_cost, curr_city, path = heapq.heappop(heap)
        if est_total_cost >= lowest_cost:
            continue
        if len(path) == size:
            if matrix[curr_city][start] != -1:
                total_cost = curr_cost + matrix[curr_city][start]
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_path = path + [start]
            continue
        for next_city in tree[curr_city]:
            if next_city not in path:
                new_cost = curr_cost + matrix[curr_city][next_city]
                unvisited = set(range(size)) - set(path) - {next_city}
                h = heuristic_func(matrix, next_city, unvisited)
                est = new_cost + h
                if est >= lowest_cost: ## jeśli koszt heurystyki przewyższa aktualny najniższy koszt to pomijamy
                    continue
                heapq.heappush(heap, (est, new_cost, next_city, path + [next_city]))
    return best_path, lowest_cost



if __name__ == "__main__":
    cities = random_cities(10)
    cities_matrix = count_distances(cities)
    print(print_distance_matrix(cities_matrix))
    print(create_tree_graph(cities_matrix))

    start = time.time()
    result_adm = astar_tsp(cities_matrix, heuristic_func=heuristic_admissible)
    print(f"\nA* admissible result: {result_adm}")
    end = time.time()
    astar_ad_exec_time = end - start

    start = time.time()
    result_inadm = astar_tsp(cities_matrix, heuristic_func=heuristic_inadmissible)
    print(f"\nA* inadmissible result: {result_inadm}")
    end = time.time()
    astar_inad_exec_time = end - start

    start = time.time()
    path_bfs, cost_bfs = bfs_salesman(cities_matrix, 0)
    print(f"\nBFS tree result: {path_bfs, cost_bfs}")
    end = time.time()
    bfs_exec_time = end - start
    #
    start = time.time()
    path_dfs, cost_dfs = dfs_salesman(cities_matrix, 0)
    print(f"\nDFS tree result: {path_dfs, cost_dfs}")
    end = time.time()
    dfs_exec_time = end - start

    start = time.time()
    print(f"\nNearest neighbour alg. result: {nearest_neigh(cities_matrix, 0)}")
    end = time.time()
    nn_exec_time = end - start
    #
    print(f"\nA* adm. execution time: {astar_ad_exec_time:.10f} seconds")
    print(f"A* inadm. execution time: {astar_inad_exec_time:.10f} seconds")
    print(f"BFS execution time: {bfs_exec_time:.10f} seconds")
    print(f"DFS execution time: {dfs_exec_time:.10f} seconds")
    print(f"Nearest Neighbor execution time: {nn_exec_time:.10f} seconds")

    # stats = {'adm': 0, 'inadm': 0}
    # for _ in range(100):
    #     start = time.time()
    #     result_adm = astar_tsp(cities_matrix, heuristic_func=heuristic_admissible)[1]
    #     # print(f"\nA* admissible result: {result_adm}")
    #     end = time.time()
    #     astar_ad_exec_time = end - start
    #
    #     start = time.time()
    #     result_inadm = astar_tsp(cities_matrix, heuristic_func=heuristic_inadmissible)[1]
    #     # print(f"\nA* inadmissible result: {result_inadm}")
    #     end = time.time()
    #     astar_inad_exec_time = end - start
    #     if astar_ad_exec_time > astar_inad_exec_time:
    #         stats['inadm'] += 1
    #     else:
    #         stats['adm'] += 1
    #
    # print(f"Faster alg: Adm {stats['adm']} - {stats['inadm']} Inadm")

