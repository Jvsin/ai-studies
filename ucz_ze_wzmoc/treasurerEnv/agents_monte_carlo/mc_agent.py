import math
import random
import copy
from collections import defaultdict
from treasure_env import MultiTreasureHunterMDP, LEFT, DOWN, RIGHT, UP
from game_display import play_game_step_by_step

# Stała eksploracji dla wzoru UCB1
EXPLORATION_CONSTANT = 1.41

class OnlineMCTSWrapper:
    """
    To jest 'oszukany' słownik. Zamiast przechowywać gotowe ruchy,
    uruchamia algorytm MCTS na żywo, gdy ktoś pyta o ruch dla danego stanu.
    """
    def __init__(self, mcts_agent):
        self.agent = mcts_agent
        
    def __contains__(self, state):
        # Zawsze mówimy "tak, znam ten stan", żeby gra nie losowała ruchu
        return True
        
    def __getitem__(self, local_state):
        # Tutaj następuje magia: gra pyta o ruch dla stanu 'local_state'.
        # My zamiast czytać z pamięci, uruchamiamy myślenie agenta.
        
        current_state_for_mcts = local_state
        
        # Korekta dla Agenta 2:
        # MCTS spodziewa się stanu "globalnego" (lub spójnego z env),
        # a game_display podaje stan lokalny (z perspektywy agenta).
        # Dla Agenta 1 to to samo. Dla Agenta 2 musimy odwrócić perspektywę.
        if self.agent.agent_id == '2':
            # Lokalny: (Ja, On, Skarby, JaTrzyma, OnTrzyma) -> (P2, P1, T, H2, H1)
            # Globalny: (P1, P2, T, H1, H2)
            p_me, p_opp, treasures, h_me, h_opp = local_state
            current_state_for_mcts = (p_opp, p_me, treasures, h_opp, h_me)
            
        return self.agent.get_action(current_state_for_mcts)

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}  # Mapa: action -> MCTSNode
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [] # Zostanie wypełnione po rozwinięciu

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=EXPLORATION_CONSTANT):
        """Wybiera najlepsze dziecko na podstawie wzoru UCB1."""
        choices_weights = []
        for action, child in self.children.items():
            if child.visits == 0:
                choices_weights.append(float('inf'))
            else:
                # Wzór UCB1: średnia wartość + eksploracja * sqrt(ln(N) / n_i)
                ucb = (child.value / child.visits) + \
                      exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(ucb)
        
        max_weight = max(choices_weights)
        best_indices = [i for i, w in enumerate(choices_weights) if w == max_weight]
        best_idx = random.choice(best_indices)
        
        actions = list(self.children.keys())
        return self.children[actions[best_idx]]

class MCTSAgent:
    def __init__(self, agent_id, env_class_ref, simulations=100, max_depth=30):
        self.agent_id = agent_id
        self.opp_id = '2' if agent_id == '1' else '1'
        self.env_ref = env_class_ref  # Referencja do klasy środowiska (do kopiowania)
        self.simulations = simulations
        self.max_depth = max_depth

    def get_action(self, current_env_state):
        """Główna metoda zwracająca najlepszą akcję dla danego stanu."""
        
        # Pobieramy stan z perspektywy TEGO agenta
        root_state = self.env_ref.get_agent_state(current_env_state, self.agent_id)
        
        # Tworzymy korzeń drzewa
        root = MCTSNode(root_state)
        root.untried_actions = self.env_ref.get_possible_actions(current_env_state, self.agent_id)

        # Jeśli nie ma akcji, zwróć cokolwiek (np. stanie w miejscu, jeśli możliwe, lub 0)
        if not root.untried_actions:
            return 0

        # Pętla symulacji MCTS
        for _ in range(self.simulations):
            node = root
            
            # Kopia środowiska do symulacji (będziemy w niej 'psuć' stan)
            sim_env = copy.deepcopy(self.env_ref)
            
            # 1. SELECTION (Wybór)
            while not node.untried_actions and node.children:
                node = node.best_child()
                # Aktualizujemy stan symulowanego środowiska, żeby pasował do węzła
                # Uwaga: W symultanicznej grze to uproszczenie, zakładamy determinizm 
                # przejścia zapisanego w drzewie dla uproszczenia obliczeń.
                
            # 2. EXPANSION (Rozwinięcie)
            if node.untried_actions:
                action = node.untried_actions.pop()
                state_after_move = self._simulate_step(sim_env, node.state, action)
                child_node = MCTSNode(state_after_move, parent=node, parent_action=action)
                
                # Uzupełnij możliwe akcje dla nowego dziecka
                # Uwaga: musimy przetłumaczyć stan lokalny na globalny dla get_possible_actions
                # lub po prostu użyć logiki mapy
                # Użyjemy metody środowiska, która radzi sobie ze stanem None (bierze z self)
                # Ale sim_env ma już zaktualizowany stan wewnętrzny po _simulate_step
                child_node.untried_actions = sim_env.get_possible_actions(None, self.agent_id)
                
                node.children[action] = child_node
                node = child_node
            
            # 3. SIMULATION (Rollout)
            rollout_depth = 0
            done = False
            total_reward = 0
            
            while not done and rollout_depth < self.max_depth:
                # Losowa akcja moja
                possible_actions = sim_env.get_possible_actions(None, self.agent_id)
                if not possible_actions: break
                my_action = random.choice(possible_actions)
                
                # Losowa akcja przeciwnika
                opp_actions = sim_env.get_possible_actions(None, self.opp_id)
                opp_action = random.choice(opp_actions) if opp_actions else 0
                
                # Krok symulacji
                if self.agent_id == '1':
                    _, rewards, done, _ = sim_env.step(my_action, opp_action)
                else:
                    _, rewards, done, _ = sim_env.step(opp_action, my_action)
                
                # Dyskontowanie nagrody (opcjonalne, ale pomocne w szukaniu najkrótszej ścieżki)
                total_reward += (0.95 ** rollout_depth) * rewards[self.agent_id]
                rollout_depth += 1
            
            # 4. BACKPROPAGATION (Wsteczna propagacja)
            while node is not None:
                node.visits += 1
                node.value += total_reward
                node = node.parent

        # Wybierz ruch z największą liczbą odwiedzin (najbardziej pewny)
        return root.best_child(exploration_weight=0).parent_action

    def _simulate_step(self, env, state_node, action):
        """
        Pomocnicza funkcja wykonująca krok w skopiowanym środowisku.
        Zakłada, że przeciwnik wykonuje losowy ruch.
        """
        opp_actions = env.get_possible_actions(None, self.opp_id)
        opp_action = random.choice(opp_actions) if opp_actions else 0
        
        if self.agent_id == '1':
            next_global_state, _, _, _ = env.step(action, opp_action)
        else:
            next_global_state, _, _, _ = env.step(opp_action, action)
            
        return env.get_agent_state(next_global_state, self.agent_id)


def train_and_play():
    # 1. Definicja mapy
    # MAP = [
    #     "A...H....",
    #     ".#.#..##.",
    #     ".H..T.#H.",
    #     ".#...H.#.",
    #     "...H....B"
    # ]
    MAP = [
        ".......",
        ".A.H.B.",
        "...#...",
        "...#...",
        "..T...."
    ]
    
    # 2. Inicjalizacja środowiska
    env = MultiTreasureHunterMDP(MAP)
    env.reset()
    print(len(env.get_all_states()))
    
    # 3. Inicjalizacja Agentów MCTS
    # Zwiększ liczbę symulacji dla lepszych wyników (np. do 500 lub 1000)
    agent1 = MCTSAgent('1', env, simulations=1000, max_depth=20)
    agent2 = MCTSAgent('2', env, simulations=1000, max_depth=20)
    
    # 4. Słowniki polityk (będą wypełniane decyzjami MCTS)
    # Kluczem jest stan (tuple), wartością akcja (int)
    policy1 = {}
    policy2 = {}
    
    print("Rozpoczynanie treningu (zbieranie polityk)...")
    training_episodes = 5  # Ilość gier treningowych
    
    for ep in range(training_episodes):
        state = env.reset()
        done = False
        step = 0
        print(f"Epizod treningowy {ep+1}/{training_episodes}")
        
        while not done and step < 500:
            # Agent 1 myśli
            action1 = agent1.get_action(state)
            state1 = env.get_agent_state(state, '1')
            policy1[state1] = action1
            
            # Agent 2 myśli
            action2 = agent2.get_action(state)
            state2 = env.get_agent_state(state, '2')
            policy2[state2] = action2
            
            # Wykonaj krok
            state, _, done, _ = env.step(action1, action2)
            step += 1
    
    print(f"Agent 1 zna {len(policy1)} stanów.")
    print(f"Agent 2 zna {len(policy2)} stanów.")
            
    print("\nTrening zakończony! Uruchamiam wizualizację...")
    
    # Ponieważ MCTS działa on-line, w słownikach policy mamy tylko stany odwiedzone
    # podczas treningu. Jeśli podczas wizualizacji trafi się nowy stan,
    # funkcja w game_display.py wylosuje akcję. 
    # Aby temu zapobiec, podmieniamy logikę w game_display lub po prostu 
    # pozwalamy MCTS działać "na żywo" (ale funkcja play_game_step_by_step przyjmuje dict).
    
    # Uruchom grę z zapisanymi decyzjami
    play_game_step_by_step(env, policy1, policy2, max_steps=50)

if __name__ == "__main__":
    train_and_play()