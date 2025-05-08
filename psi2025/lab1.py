import math
import random
from collections import deque


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

    num_to_cut = int((size**2 - size) * cut_connection)
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


def print_graph(matrix):
    graph = {i: {} for i in range(len(cities))}
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                graph[i][j] = matrix[i][j]
    for city, neighbours in graph.items():
        print(f"\nCity {city}: {neighbours}")


def create_tree_graph(cities_matrix):
    graph = {}
    for i in range(len(cities_matrix)):
        city_routes = []
        for j in range(len(cities_matrix)):
            if cities_matrix[i][j] != -1 and i != j:
                city_routes.append(j)
        graph[i] = city_routes
    print(f"\nCities graph: {graph}")
    return graph


def bfs_salesman(cities_matrix, start=0):
    best_path = None
    lowest_cost = float('inf')

    tree = create_tree_graph(cities_matrix)
    queue = deque([(start, [start], 0)])
    while queue:
        curr_city, path, cost = queue.popleft()
        print(curr_city, path, cost)

        if len(path) == len(cities_matrix):
            if start in tree[curr_city]:
                cost += cities_matrix[curr_city][start]
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path + [start]
                    print(f"Found best way: {best_path} with cost: {cost}")
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
        print(curr_city, path, cost)

        if len(path) == len(cities_matrix):
            if start in tree[curr_city]:
                cost += cities_matrix[curr_city][start]
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path + [start]
                    print(f"Found best way: {best_path} with cost: {cost}")
            continue

        for next_city in reversed(tree[curr_city]):
            if next_city not in path:
                stack.append((next_city, path + [next_city], cost + cities_matrix[curr_city][next_city]))

    return best_path, lowest_cost


if __name__ == "__main__":
    cities = random_cities(4)

    distance_matrix = count_distances(cities)
    print_distance_matrix(distance_matrix)

    print("\nBFS tree search: ..........................")
    path_bfs, cost_bfs = bfs_salesman(distance_matrix, 0)

    print("\nDFS tree search: ..........................")
    path_dfs, cost_dfs = dfs_salesman(distance_matrix, 0)

    print(f"BFS Final best way: {path_bfs}, cost: {cost_bfs}")
    print(f"DFS Final best way: {path_dfs}, cost: {cost_dfs}")
