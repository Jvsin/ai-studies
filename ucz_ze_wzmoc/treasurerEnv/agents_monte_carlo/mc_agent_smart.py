import math
import random
import copy
from treasure_env import MultiTreasureHunterMDP, ACTIONS

# Stała eksploracji (balans między sprawdzaniem nowego a wybieraniem pewnego)
EXPLORATION_CONSTANT = 1.41

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None, untried_actions=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions if untried_actions is not None else []

    def best_child(self, exploration_weight=EXPLORATION_CONSTANT):
        choices_weights = []
        for action, child in self.children.items():
            if child.visits == 0:
                choices_weights.append(float('inf'))
            else:
                # Wzór UCB1
                ucb = (child.value / child.visits) + \
                      exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(ucb)
        
        if not choices_weights:
            return None # Nie powinno się zdarzyć

        max_weight = max(choices_weights)
        best_indices = [i for i, w in enumerate(choices_weights) if w == max_weight]
        best_idx = random.choice(best_indices)
        actions = list(self.children.keys())
        return self.children[actions[best_idx]]

class MCTSAgent:
    def __init__(self, agent_id, env_template, simulations=1000, max_depth=60):
        self.agent_id = agent_id
        self.opp_id = '2' if agent_id == '1' else '1'
        # Tworzymy jedno środowisko robocze do symulacji (nie kopiujemy go w pętli!)
        self.sim_env = MultiTreasureHunterMDP(env_template.original_map) 
        self.simulations = simulations
        self.max_depth = max_depth

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_smart_rollout_action(self, env, agent_id, epsilon=0.2):
        """
        Szybka polityka do symulacji (Rollout).
        Unika ścian i dziur, kieruje się z grubsza w stronę celu.
        """
        # Pobieramy możliwe akcje (bez wychodzenia za mapę i w ściany)
        # UWAGA: env.get_possible_actions używa aktualnego stanu self.sim_env
        possible_actions = env.get_possible_actions(None, agent_id)
        
        if not possible_actions:
            return 0 # Brak ruchu
            
        # 1. Trochę losowości
        if random.random() < epsilon:
            return random.choice(possible_actions)

        # 2. Pobierz stan
        p1 = env.agent_pos['1']
        p2 = env.agent_pos['2']
        treasures = env.treasures
        h1 = env.agent_holding['1']
        h2 = env.agent_holding['2']
        
        current_pos = p1 if agent_id == '1' else p2
        is_holding = h1 if agent_id == '1' else h2
        
        # 3. Wyznacz Cel
        target_pos = None
        if is_holding:
            # Jeśli mam skarb -> Idź do bazy
            if agent_id in env.bases:
                target_pos = env.bases[agent_id]
        else:
            # Jeśli nie mam skarbu -> Idź do najbliższego
            if treasures:
                min_dist = float('inf')
                for t_pos in treasures:
                    dist = self._manhattan_dist(current_pos, t_pos)
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = t_pos
            else:
                # Brak skarbów na mapie - goń przeciwnika (jeśli on ma skarb) lub idź do bazy
                opp_holding = h2 if agent_id == '1' else h1
                if opp_holding:
                    target_pos = p2 if agent_id == '1' else p1 # Goń przeciwnika
                else:
                    target_pos = env.bases[agent_id] # Wracaj

        if target_pos is None:
            return random.choice(possible_actions)

        # 4. Wybierz ruch skracający dystans, ALE bezpieczny
        best_action = random.choice(possible_actions)
        min_dist_to_target = float('inf')
        
        x, y = current_pos
        
        # Przetasuj akcje, żeby nie faworyzować jednego kierunku przy remisach
        random.shuffle(possible_actions)

        for action in possible_actions:
            dx, dy = ACTIONS[action]
            next_x, next_y = x + dx, y + dy
            
            # Sprawdź czy to nie dziura (Heurystyka bezpieczeństwa)
            # W rolloutach unikamy dziur za wszelką cenę, chyba że epsilon wylosuje inaczej
            if env.map[next_y][next_x] == 'H':
                continue

            dist = self._manhattan_dist((next_x, next_y), target_pos)
            
            if dist < min_dist_to_target:
                min_dist_to_target = dist
                best_action = action
                
        return best_action

    def get_action(self, current_state_tuple):
        """Główna pętla MCTS"""
        
        # 1. Setup korzenia
        # Musimy wiedzieć, jakie akcje są dostępne w obecnym stanie
        self.sim_env.set_state(current_state_tuple)
        available_actions = self.sim_env.get_possible_actions(None, self.agent_id)
        
        # Korzeń drzewa reprezentuje AKTUALNY stan gry
        root = MCTSNode(state=current_state_tuple, untried_actions=list(available_actions))

        if not available_actions:
            return 0

        # 2. Pętla symulacji
        for _ in range(self.simulations):
            node = root
            
            # Reset środowiska symulacyjnego do stanu korzenia
            self.sim_env.set_state(root.state)

            # --- A. SELECTION (Schodzenie w dół drzewa) ---
            # Idziemy w dół dopóki węzeł jest w pełni rozwinięty (brak untried) i ma dzieci
            while not node.untried_actions and node.children:
                node = node.best_child()
                # Wykonujemy ten ruch w symulatorze, żeby zaktualizować stan
                # Uwaga: w drzewie zapisujemy stan PO ruchu, więc wystarczy set_state
                self.sim_env.set_state(node.state)

            # --- B. EXPANSION (Dodanie nowego węzła) ---
            if node.untried_actions:
                action = node.untried_actions.pop()
                
                # Symulujemy ruch przeciwnika (zakładamy, że jest w miarę mądry)
                opp_action = self._get_smart_rollout_action(self.sim_env, self.opp_id, epsilon=0.4)
                
                # Wykonaj krok w symulacji
                if self.agent_id == '1':
                    _, _, done, _ = self.sim_env.step(action, opp_action)
                else:
                    _, _, done, _ = self.sim_env.step(opp_action, action)
                
                # Pobierz nowy stan po ruchu
                new_state = self.sim_env._get_state()
                
                # Sprawdź dostępne akcje dla nowego stanu
                next_actions = self.sim_env.get_possible_actions(None, self.agent_id) if not done else []
                
                child_node = MCTSNode(state=new_state, parent=node, parent_action=action, untried_actions=next_actions)
                node.children[action] = child_node
                node = child_node
            
            # --- C. SIMULATION / ROLLOUT (Gdybanie do przodu) ---
            rollout_depth = 0
            done = self.sim_env.is_terminal(self.sim_env._get_state()) # Czy już koniec?
            total_reward = 0
            
            # Zapobiegamy pętlom w rolloucie (żeby nie chodził w kółko)
            # Ale w prostym MCTS wystarczy limit głębokości
            
            while not done and rollout_depth < self.max_depth:
                my_action = self._get_smart_rollout_action(self.sim_env, self.agent_id)
                opp_action = self._get_smart_rollout_action(self.sim_env, self.opp_id)
                
                if self.agent_id == '1':
                    _, rewards, done, _ = self.sim_env.step(my_action, opp_action)
                else:
                    _, rewards, done, _ = self.sim_env.step(opp_action, my_action)
                
                # Dyskontowanie nagród
                total_reward += (0.95 ** rollout_depth) * rewards[self.agent_id]
                rollout_depth += 1
            
            # --- D. BACKPROPAGATION (Aktualizacja) ---
            while node is not None:
                node.visits += 1
                node.value += total_reward
                node = node.parent

        # Wybierz najlepszy ruch (najwięcej odwiedzin = najbardziej pewny)
        # exploration_weight=0 wyłącza czynnik losowy eksploracji
        best_child = root.best_child(exploration_weight=0)
        
        if best_child is None:
            return random.choice(available_actions)
            
        return best_child.parent_action

class OnlineMCTSWrapper:
    def __init__(self, agent):
        self.agent = agent
    def __contains__(self, state):
        return True
    def __getitem__(self, state):
        # Stan przekazywany przez game_display jest już globalny lub lokalny zależnie od implementacji
        # W Twoim game_display: env.get_agent_state() zwraca tuple.
        # Agent MCTS spodziewa się stanu w formacie (p1, p2, treasures, h1, h2).
        # Upewniamy się, że to co dostajemy pasuje do logiki environment.
        return self.agent.get_action(state)

def run_online():
    from game_display import play_game_step_by_step
    
    # MAP = [
    #     "A....H...T",
    #     ".#.#...##.",
    #     ".H.#.T.#H.",
    #     ".##...H.#.",
    #     "T...H....B"
    # ]
    MAP = [
        ".A...B.",
        "#..#..#",
        "H..#..H",
        "#..H..#",
        ".T...T."
    ]
    
    env = MultiTreasureHunterMDP(MAP)
    
    # Zwiększamy głębokość symulacji, żeby agent widział odległy cel!
    # 1000 symulacji, głębokość 60
    print("Inicjalizacja MCTS (Sims=1000, Depth=60)...")
    
    agent1 = MCTSAgent('1', env, simulations=1000, max_depth=60)
    agent2 = MCTSAgent('2', env, simulations=1000, max_depth=60)
    
    policy1 = OnlineMCTSWrapper(agent1)
    policy2 = OnlineMCTSWrapper(agent2)
    
    play_game_step_by_step(env, policy1, policy2, max_steps=100)

if __name__ == "__main__":
    run_online()