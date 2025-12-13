"""
Monte Carlo Tree Search Online dla środowiska TreasureHunter
Dwóch agentów MCTS uczących się online w grze
"""
import math
import random
import copy
import time
from treasure_env import MultiTreasureHunterMDP

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


class MCTSNode:
    """Węzeł drzewa MCTS"""
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action #action wykonana by tu przyjść
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        
    def is_fully_expanded(self):
        if self.untried_actions is None:
            return False
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.41):
        """UCB1"""
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                weight = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                weight = exploitation + exploration
            choices_weights.append(weight)
        
        return self.children[choices_weights.index(max(choices_weights))]
    
    def most_visited_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value / c.visits if c.visits else 0)


class MCTSOnlineAgent:
    """Agent MCTS Online"""
    def __init__(self, env, agent_id='1', num_simulations=100, max_depth=10):
        self.env = env
        self.agent_id = agent_id
        self.opponent_id = '2' if agent_id == '1' else '1'
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        
    def select_action(self, state):
        """Wybiera akcję za pomocą MCTS"""
        root = MCTSNode(state)
        agent_state = self.env.get_agent_state(state, self.agent_id)
        root.untried_actions = self.env.get_possible_actions(agent_state, self.agent_id)
        
        if not root.untried_actions:
            return 0
        
        for _ in range(self.num_simulations):
            node = root
            sim_state = state
            
            # Selection
            while not self.env.is_terminal(sim_state) and node.is_fully_expanded():
                node = node.best_child()
                sim_state = self._simulate_action(sim_state, node.action)[0]
            
            # Expansion
            if not self.env.is_terminal(sim_state) and node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, reward = self._simulate_action(sim_state, action)
                
                child = MCTSNode(next_state, parent=node, action=action)
                agent_state_child = self.env.get_agent_state(next_state, self.agent_id)
                child.untried_actions = self.env.get_possible_actions(agent_state_child, self.agent_id)
                node.children.append(child)
                node = child
                sim_state = next_state
            
            # Simulation (Rollout)
            rollout_reward = self._rollout(sim_state)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += rollout_reward
                node = node.parent
        
        # best_child = root.most_visited_child()
        # return best_child.action if best_child else root.untried_actions[0]
        if self.agent_id == '1':
            pass
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def _simulate_action(self, state, my_action):
        """Symuluje akcję"""
        opponent_action = self._opponent_policy(state)
        
        if self.agent_id == '1':
            action1, action2 = my_action, opponent_action
        else:
            action1, action2 = opponent_action, my_action
        
        temp_env = self._create_temp_env(state)
        next_state, rewards, done, info = temp_env.step(action1, action2)
        return next_state, rewards[self.agent_id]
    
    def _rollout(self, state, depth=0):
        """Rollout z heurystyką - agent zmierza do celu"""
        total_reward = 0.0
        discount = 1.0
        current_state = state
        current_depth = depth
        
        while not self.env.is_terminal(current_state) and current_depth < self.max_depth:
            agent_state = self.env.get_agent_state(current_state, self.agent_id)
            possible_actions = self.env.get_possible_actions(agent_state, self.agent_id)
            
            if not possible_actions:
                break
            
            # Rozpakuj stan z perspektywy agenta
            my_pos, opponent_pos, treasures, my_holding, opponent_holding = agent_state
            my_base = self.env.bases[self.agent_id]
            
            # Heurystyka: określ cel
            if my_holding:
                # Mam skarb -> idź do bazy
                target = my_base
            elif treasures:
                # Nie mam skarbu -> idź do najbliższego skarbu
                target = min(treasures, key=lambda t: abs(t[0] - my_pos[0]) + abs(t[1] - my_pos[1]))
            else:
                # Brak skarbów -> losowy ruch
                my_action = random.choice(possible_actions)
                next_state, reward = self._simulate_action(current_state, my_action)
                total_reward += discount * reward
                current_state = next_state
                current_depth += 1
                discount *= 0.99
                continue
            
            # Wybierz akcję która zbliża do celu (Manhattan distance)
            best_action = possible_actions[0]
            best_dist = float('inf')
            
            for action in possible_actions:
                dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
                new_pos = (my_pos[0] + dx, my_pos[1] + dy)
                dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            
            # Symuluj wybraną akcję
            next_state, reward = self._simulate_action(current_state, best_action)
            total_reward += discount * reward
            current_state = next_state
            current_depth += 1
            discount *= 0.99
        
        return total_reward
    
    def _opponent_policy(self, state):
        """Prosta polityka przeciwnika"""
        # Użyj przekonwertowanego stanu z perspektywy przeciwnika
        opponent_state = self.env.get_agent_state(state, self.opponent_id)
        possible_actions = self.env.get_possible_actions(opponent_state, self.opponent_id)
        
        if not possible_actions:
            return 0
        
        # opponent_state jest już z perspektywy przeciwnika
        # (opponent_pos, my_pos, treasures, opponent_hold, my_hold)
        opponent_pos, _, treasures, opponent_holding, _ = opponent_state
        opponent_base = self.env.bases[self.opponent_id]
        
        if opponent_holding:
            target = opponent_base
        elif treasures:
            target = min(treasures, key=lambda t: abs(t[0] - opponent_pos[0]) + abs(t[1] - opponent_pos[1]))
        else:
            return random.choice(possible_actions)
        
        best_action = possible_actions[0]
        best_dist = float('inf')
        
        for action in possible_actions:
            dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
            new_pos = (opponent_pos[0] + dx, opponent_pos[1] + dy)
            dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action
    
    def _create_temp_env(self, state):
        """Tworzy tymczasowe środowisko"""
        temp_env = copy.deepcopy(self.env)
        pos1, pos2, treasures, hold1, hold2 = state
        temp_env.agent_pos = {'1': pos1, '2': pos2}
        temp_env.treasures = set(treasures)
        temp_env.agent_holding = {'1': hold1, '2': hold2}
        return temp_env


def render_game_state(env, step_num, state, action1=None, action2=None, rewards=None):
    """Renderuje stan gry"""
    print("\n" + "-"*70)
    print(f"KROK {step_num}")
    print("-"*70)
    
    grid = [row[:] for row in env.map]
    
    for x, y in env.treasures:
        grid[y][x] = 'T'
    
    for agent_id, (bx, by) in env.bases.items():
        if grid[by][bx] == '.':
            grid[by][bx] = 'A' if agent_id == '1' else 'B'
    
    p1 = env.agent_pos['1']
    p2 = env.agent_pos['2']
    
    if p1 == p2:
        grid[p1[1]][p1[0]] = 'X'
    else:
        grid[p1[1]][p1[0]] = '①' if env.agent_holding['1'] else '1'
        grid[p2[1]][p2[0]] = '②' if env.agent_holding['2'] else '2'
    
    print("\nPlansza:")
    for row in grid:
        print("  " + ' '.join(row))
    
    print(f"\nStatus:")
    print(f"  Agent 1 - Pozycja: {p1}, Trzyma skarb: {env.agent_holding['1']}, Punkty: {env.agent_score['1']}")
    print(f"  Agent 2 - Pozycja: {p2}, Trzyma skarb: {env.agent_holding['2']}, Punkty: {env.agent_score['2']}")
    print(f"  Skarby na mapie: {len(env.treasures)}")
    
    if action1 is not None and action2 is not None:
        print(f"\nWykonane akcje:")
        print(f"  Agent 1: {ACTION_NAMES[action1]}")
        print(f"  Agent 2: {ACTION_NAMES[action2]}")
        
        if rewards:
            print(f"\nOtrzymane nagrody:")
            print(f"  Agent 1: {rewards['1']:+.1f}")
            print(f"  Agent 2: {rewards['2']:+.1f}")


def play_mcts_vs_mcts(env, num_simulations=100, max_steps=100):
    """Gra MCTS vs MCTS"""
    agent1 = MCTSOnlineAgent(env, '1', num_simulations=num_simulations)
    agent2 = MCTSOnlineAgent(env, '2', num_simulations=num_simulations)
    
    state = env.reset()
    render_game_state(env, 0, state)
    
    print("\nRozpoczynanie gry...")
    time.sleep(2)
    
    total_reward_1 = 0
    total_reward_2 = 0
    
    for step in range(1, max_steps + 1):
        print(f"\n[Agent 1] Planowanie ruchu...")
        action1 = agent1.select_action(state)
        
        print(f"[Agent 2] Planowanie ruchu...")
        action2 = agent2.select_action(state)
        
        state, rewards, done, info = env.step(action1, action2)
        total_reward_1 += rewards['1']
        total_reward_2 += rewards['2']
        
        render_game_state(env, step, state, action1, action2, rewards)
        
        if info:
            if info.get('collision'):
                print("\n  ⚠ KOLIZJA! Obaj agenci wrócili do baz!")
            if info.get('1_pick'):
                print("\n  ✓ Agent 1 podniósł skarb!")
            if info.get('2_pick'):
                print("\n  ✓ Agent 2 podniósł skarb!")
            if info.get('1_deposit'):
                print("\n  ★ Agent 1 zdeponował skarb! +1 punkt")
            if info.get('2_deposit'):
                print("\n  ★ Agent 2 zdeponował skarb! +1 punkt")
            if info.get('1_trap'):
                print("\n  ✗ Agent 1 wpadł w pułapkę!")
            if info.get('2_trap'):
                print("\n  ✗ Agent 2 wpadł w pułapkę!")
        
        if done:
            print("\n" + "="*70)
            print("GRA ZAKOŃCZONA!")
            print("="*70)
            print(f"\nCałkowite nagrody:")
            print(f"  Agent 1: {total_reward_1:.1f}")
            print(f"  Agent 2: {total_reward_2:.1f}")
            print(f"\nKońcowe punkty:")
            print(f"  Agent 1: {env.agent_score['1']}")
            print(f"  Agent 2: {env.agent_score['2']}")
            
            if env.agent_score['1'] > env.agent_score['2']:
                print(f"\n🏆 ZWYCIĘZCA: Agent 1!")
            elif env.agent_score['2'] > env.agent_score['1']:
                print(f"\n🏆 ZWYCIĘZCA: Agent 2!")
            else:
                print(f"\n🤝 REMIS!")
            print("="*70)
            break

        # input("Enter by kontynuować...")
        time.sleep(1)
    
    if not done:
        print("\n" + "="*70)
        print("OSIĄGNIĘTO MAKSYMALNĄ LICZBĘ KROKÓW")
        print("="*70)


if __name__ == "__main__":
    # map_lines = [
    #     "A.....#",
    #     "###...H",
    #     "#..T..#",
    #     "H...###",
    #     "#.....B",
    # ]
    map_lines = [
        "A...##",
        "###..H",
        "#.#T.#",
        "H...##",
        "##...B",
    ]
    
    print("="*70)
    print("MONTE CARLO TREE SEARCH ONLINE")
    print("Dwóch agentów MCTS uczących się w grze")
    print("="*70)
    
    print("\nMapa:")
    for line in map_lines:
        print("  " + line)
    
    print("\nLegenda:")
    print("  A, B - Bazy agentów")
    print("  T - Skarby")
    print("  H - Pułapki")
    print("  # - Ściany")
    
    env = MultiTreasureHunterMDP(map_lines)
    
    # Uruchom grę
    play_mcts_vs_mcts(env, num_simulations=500, max_steps=100)
