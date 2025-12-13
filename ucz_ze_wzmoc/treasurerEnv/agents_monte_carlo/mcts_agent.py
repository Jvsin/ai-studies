import math
import random
from treasure_env import LEFT, DOWN, RIGHT, UP

# Stałe konfiguracyjne dla MCTS
ITERATIONS = 1000  # Liczba symulacji na ruch (im więcej, tym mądrzejszy agent)
ROLLOUT_DEPTH = 50 # Jak daleko w przyszłość symulujemy losowo
EXPLORATION_CONSTANT = 1.41

class MonteCarloNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Akcja, która doprowadziła do tego stanu
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        # Wzór UCT (Upper Confidence Bound applied to Trees)
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

def dist_manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_objective_distance(env, state, agent_id):
    """
    Funkcja heurystyczna.
    Jeśli agent trzyma skarb -> odległość do bazy.
    Jeśli nie trzyma -> odległość do najbliższego skarbu.
    """
    pos1, pos2, treasures, hold1, hold2 = state
    
    current_pos = pos1 if agent_id == '1' else pos2
    is_holding = hold1 if agent_id == '1' else hold2
    base_pos = env.bases[agent_id]

    if is_holding:
        return dist_manhattan(current_pos, base_pos)
    else:
        if not treasures:
            # Jeśli nie ma skarbów na mapie, a agent nie trzyma,
            # może iść do bazy przeciwnika (chase) lub czekać.
            # Tutaj upraszczamy: idź do środka mapy lub bazy.
            return dist_manhattan(current_pos, (env.width//2, env.height//2))
        
        # Znajdź najbliższy skarb
        min_dist = float('inf')
        for t in treasures:
            d = dist_manhattan(current_pos, t)
            if d < min_dist:
                min_dist = d
        return min_dist

def smart_rollout(env, start_state, agent_id):
    """
    Symulacja rozgrywki od danego stanu w głąb.
    Zwraca szacowaną nagrodę (wartość stanu).
    """
    current_state = start_state
    
    # Sprawdzenie czy stan jest terminalny (wygrana)
    if env.is_terminal(current_state):
        return 1.0

    steps = 0
    total_reward = 0
    
    # Symulujemy ruchy tylko naszego agenta (zakładamy, że przeciwnik stoi w miejscu
    # lub jest częścią środowiska w get_next_states, co upraszcza obliczenia w rollout)
    while steps < ROLLOUT_DEPTH:
        steps += 1
        
        actions = env.get_possible_actions(current_state, agent_id)
        if not actions:
            break

        # Polityka epsilon-greedy w symulacji:
        # 80% szans na ruch przybliżający do celu, 20% losowy
        if random.random() < 0.8:
            best_a = actions[0]
            min_dist = float('inf')
            # Sprawdzamy, która akcja przybliża nas do celu
            for a in actions:
                next_s = env.get_next_states(current_state, a, agent_id)[0]
                d = get_objective_distance(env, next_s, agent_id)
                if d < min_dist:
                    min_dist = d
                    best_a = a
            action = best_a
        else:
            action = random.choice(actions)

        # Pobieramy następny stan (symulacja ruchu)
        # get_next_states zwraca listę, bierzemy [0] (deterministyczne środowisko w tym aspekcie)
        next_state = env.get_next_states(current_state, action, agent_id)[0]
        
        # Prosta nagroda za dotarcie do bazy ze skarbem w trakcie rolloutu
        # Analizujemy zmianę stanu
        _, _, old_treasures, old_hold, _ = current_state if agent_id == '1' else (current_state[1], current_state[0], current_state[2], current_state[4], current_state[3])
        _, _, new_treasures, new_hold, _ = next_state if agent_id == '1' else (next_state[1], next_state[0], next_state[2], next_state[4], next_state[3])
        
        # Nagroda za podniesienie
        if not old_hold and new_hold:
            total_reward += 0.3
        # Nagroda za oddanie (duża)
        if old_hold and not new_hold and len(new_treasures) == len(old_treasures): 
            # (To uproszczony warunek 'deposit', bo treasures się nie zmieniło, a hold zniknął przy bazie)
            # W treasure_env logika deposit jest: hold -> False, score++
            # Musimy sprawdzić pozycję bazy
            pos = next_state[0] if agent_id == '1' else next_state[1]
            if pos == env.bases[agent_id]:
                return 1.0 # Znaleźliśmy zwycięską ścieżkę w symulacji!

        current_state = next_state

    # Heurystyczna ocena stanu końcowego symulacji
    # Im bliżej celu, tym lepiej. Normalizujemy do [0, 1]
    final_dist = get_objective_distance(env, current_state, agent_id)
    max_dist = env.width + env.height
    heuristic_score = 1.0 - (min(final_dist, max_dist) / max_dist)
    
    return (total_reward + heuristic_score) / (1 + total_reward)

def mcts_policy(env, root_state, agent_id):
    """
    Główna funkcja wywoływana, by zdecydować o ruchu.
    """
    root = MonteCarloNode(root_state)
    root.untried_actions = env.get_possible_actions(root_state, agent_id)

    for _ in range(ITERATIONS):
        node = root
        temp_state = root_state

        # 1. SELECTION
        while not node.untried_actions and node.children:
            node = node.best_child()
            temp_state = node.state

        # 2. EXPANSION
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            
            # Symulujemy stan po wykonaniu akcji
            next_state_list = env.get_next_states(temp_state, action, agent_id)
            new_state = next_state_list[0] # Zakładamy determinizm
            
            child_node = MonteCarloNode(new_state, parent=node, action=action)
            child_node.untried_actions = env.get_possible_actions(new_state, agent_id)
            node.children.append(child_node)
            node = child_node
            temp_state = new_state

        # 3. SIMULATION (ROLLOUT)
        simulation_result = smart_rollout(env, temp_state, agent_id)

        # 4. BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.value += simulation_result
            node = node.parent

    # Wybór ruchu o największej liczbie odwiedzin (najbardziej pewny)
    if not root.children:
        # Jeśli z jakiegoś powodu nie ma dzieci (np. brak ruchów), losuj
        possible = env.get_possible_actions(root_state, agent_id)
        return random.choice(possible) if possible else 0 # 0 = STAY/LEFT fallback

    best_move_node = max(root.children, key=lambda c: c.visits)
    return best_move_node.action