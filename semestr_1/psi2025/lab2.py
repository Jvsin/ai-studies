import math
import random
from collections import deque
from itertools import permutations
import time


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


def farthest_neighbor(cities_matrix, start=0):
    size = len(cities_matrix)
    path = [start]
    cost = 0
    current_city = start

    visited = set(path)

    while len(path) < size:
        max_dist = -1
        next_city = None
        for city in range(size):
            if city not in visited and cities_matrix[current_city][city] > max_dist:
                max_dist = cities_matrix[current_city][city]
                next_city = city

        if next_city is None:
            print("Brak możliwej ścieżki.")
            return None, 0

        path.append(next_city)
        cost += cities_matrix[current_city][next_city]
        visited.add(next_city)
        current_city = next_city

    if cities_matrix[current_city][start] != -1:
        cost += cities_matrix[current_city][start]
        path.append(start)
    else:
        print("No possible path")
        return None, 0

    return path, cost


if __name__ == "__main__":
    cities = random_cities(4)
    cities_matrix = count_distances(cities)
    print(print_distance_matrix(cities_matrix))
    print(create_tree_graph(cities_matrix))

    start = time.time()
    print(f"\nNearest neighbour alg. result: {nearest_neigh(cities_matrix, 0)}")
    end = time.time()
    nn_exec_time = end - start

    start = time.time()
    print(f"Farthest neighbour alg. result: {farthest_neighbor(cities_matrix, 0)}")
    end = time.time()
    fn_exec_time = end - start

    start = time.time()
    path_bfs, cost_bfs = bfs_salesman(cities_matrix, 0)
    print(f"BFS tree result: {path_bfs, cost_bfs}")
    end = time.time()
    bfs_exec_time = end - start

    start = time.time()
    path_dfs, cost_dfs = dfs_salesman(cities_matrix, 0)
    print(f"DFS tree result: {path_dfs, cost_dfs}")
    end = time.time()
    dfs_exec_time = end - start

    print(f"\nNearest Neighbor execution time: {nn_exec_time:.10f} seconds")
    print(f"Farthest Neighbor execution time: {fn_exec_time:.10f} seconds")
    print(f"BFS execution time: {bfs_exec_time:.10f} seconds")
    print(f"DFS execution time: {dfs_exec_time:.10f} seconds\n")
