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


def aco_tsp(cities_matrix, n_ants, n_iterations, alpha=1, beta=2, rho=0.5, Q=100):
    n = len(cities_matrix)
    graph = create_tree_graph(cities_matrix)

    pheromones = np.zeros((n, n))
    for i in range(n):
        for j in graph[i]:
            pheromones[i][j] = 0.1

    best_tour = None
    best_cost = float('inf')

    for _ in range(n_iterations):
        tours = []
        costs = []

        for _ in range(n_ants):
            tour = [np.random.randint(n)]

            while len(tour) < n:
                current_city = tour[-1]
                unvisited = [j for j in graph[current_city] if j not in tour]
                if not unvisited:
                    break
                probs = []
                for v in unvisited:
                    cost = cities_matrix[current_city][v]
                    prob = (pheromones[current_city][v] ** alpha) * ((1 / cost) ** beta)
                    probs.append(prob)
                if sum(probs) == 0:
                    break
                probs = np.array(probs) / sum(probs)
                next_city = int(np.random.choice(unvisited, p=probs))
                tour.append(next_city)

            if len(tour) == n:
                if tour[0] in graph[tour[-1]]:
                    tour.append(tour[0])
                    final_cost = sum(cities_matrix[tour[i]][tour[i + 1]] for i in range(n))
                    if final_cost < best_cost:
                        best_cost = final_cost
                        best_tour = tour
                    tours.append(tour)
                    costs.append(final_cost)

        pheromones *= rho
        for tour, cost in zip(tours, costs):
            for i in range(n):
                pheromones[tour[i]][tour[i + 1]] += Q / cost

    return best_tour, best_cost, best_tour[0]


if __name__ == "__main__":
    cities = random_cities(5)
    cities_matrix = count_distances(cities)
    print(print_distance_matrix(cities_matrix))
    print(create_tree_graph(cities_matrix))

    start = time.time()
    path_aco, cost_aco, city_start = aco_tsp(cities_matrix, n_ants=10, n_iterations=10)
    end = time.time()
    print(f"\nACO alg result: {path_aco, cost_aco}")
    print(f"Execution time: {(end - start):.10f}")

    start = time.time()
    path_bfs, cost_bfs = bfs_salesman(cities_matrix, start=city_start)
    end = time.time()
    print(f"\nBFS tree result: {path_bfs, cost_bfs}")
    print(f"Execution time: {(end - start):.10f}")

    start = time.time()
    path_dfs, cost_dfs = dfs_salesman(cities_matrix, start=city_start)
    print(f"\nDFS tree result: {path_dfs, cost_dfs}")
    end = time.time()
    print(f"Execution time: {(end - start):.10f}")


