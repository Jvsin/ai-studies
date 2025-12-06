from treasure_env import MultiTreasureHunterMDP as MTH
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAP = [
    "A....H...T",
    ".#.#...##.",
    ".H.#T..#H.",
    ".##...H.#.",
    "T...H....B"
]

def get_agents_states(mdp, agent='1'):
    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state] = 0
        actions = mdp.get_possible_actions(current_state, agent)
        policy[current_state] = actions[0]
    
    return V, policy


def value_iteration(mdp, gamma, theta, agent='1'):
    """
            This function calculate optimal policy for the specified MDP using Value Iteration approach:

            'mdp' - model of the environment, use following functions:
                get_all_states - return list of all states available in the environment
                get_possible_actions - return list of possible actions for the given state
                get_next_states - return list of possible next states with a probability for transition from state by taking
                                  action into next_state
                get_reward - return the reward after taking action in state and landing on next_state


            'gamma' - discount factor for MDP
            'theta' - algorithm should stop when minimal difference between previous evaluation of policy and current is
                      smaller than theta
            Function returns optimal policy and value function for the policy
       """
    V = dict()
    policy = dict()

    # init with a policy with first avail action for each state
    for current_state in mdp.get_all_states():
        V[current_state] = 0
        actions = mdp.get_possible_actions(current_state, agent)
        policy[current_state] = actions[0]
    
    iter_counter = 0
    while True:
        delta = 0
        for s in mdp.get_all_states():
            last_v = V[s]
            possible_actions = mdp.get_possible_actions(s, agent)
            a_values = dict()

            for a in possible_actions:
                p_next_states = mdp.get_next_states(s, a)
                result = 0
                for ns in p_next_states:
                    reward = mdp.get_reward(s, a, ns, agent)
                    # result += p_next_states[ns] * (reward + gamma * V[ns])
                    result += 1 * (reward + gamma * V[ns])
                a_values[a] = result

            best_action = max(a_values, key=a_values.get)
            V[s] = a_values[best_action]
            policy[s] = best_action 
            delta = max(delta, abs(last_v - V[s]))
        
        # if iter_counter % 10:
        #     print(f"Wymieliłem {iter_counter} iteracje")
        
        if delta < theta:
            break

        iter_counter += 1

    print(f"Zakończono po {iter_counter} iteracjach.")
    return policy, V


def render_game_state(env, step_num, action1=None, action2=None, rewards=None):
    """
    Renderuje aktualny stan gry w konsoli.
    """
    ACTION_NAMES = {0: "LEFT ←", 1: "DOWN ↓", 2: "RIGHT →", 3: "UP ↑"}
    
    print("\n" + "-"*70)
    print(f"KROK {step_num}")
    print("-"*70)
    
    # Przygotuj siatkę do wyświetlenia
    grid = [row[:] for row in env.map]
    
    # Dodaj skarby
    for x, y in env.treasures:
        grid[y][x] = 'T'
    
    # Dodaj bazy
    for agent_id, (bx, by) in env.bases.items():
        if grid[by][bx] == '.':
            grid[by][bx] = 'A' if agent_id == '1' else 'B'
    
    # Dodaj agentów (na końcu, żeby byli na wierzchu)
    p1 = env.agent_pos['1']
    p2 = env.agent_pos['2']
    
    if p1 == p2:
        grid[p1[1]][p1[0]] = 'X'  # Kolizja
    else:
        if env.agent_holding['1']:
            grid[p1[1]][p1[0]] = '①'  # Agent 1 z skarbem
        else:
            grid[p1[1]][p1[0]] = '1'
        
        if env.agent_holding['2']:
            grid[p2[1]][p2[0]] = '②'  # Agent 2 z skarbem
        else:
            grid[p2[1]][p2[0]] = '2'
    
    # Wyświetl siatkę
    print("\nPlansza:")
    for row in grid:
        print("  " + ' '.join(row))
    
    # Status
    print(f"\nStatus:")
    print(f"  Agent 1 - Pozycja: {p1}, Trzyma skarb: {env.agent_holding['1']}, Punkty: {env.agent_score['1']}")
    print(f"  Agent 2 - Pozycja: {p2}, Trzyma skarb: {env.agent_holding['2']}, Punkty: {env.agent_score['2']}")
    print(f"  Skarby na mapie: {len(env.treasures)}")
    
    # Akcje i nagrody
    if action1 is not None and action2 is not None:
        print(f"\nWykonane akcje:")
        print(f"  Agent 1: {ACTION_NAMES[action1]}")
        print(f"  Agent 2: {ACTION_NAMES[action2]}")
        
        if rewards:
            print(f"\nOtrzymane nagrody:")
            print(f"  Agent 1: {rewards['1']:+.1f}")
            print(f"  Agent 2: {rewards['2']:+.1f}")


def play_game_step_by_step(env, policy1, policy2, max_steps=100):
    """
    Rozgrywa grę krok po kroku, wyświetlając stan po każdym ruchu.
    
    Args:
        env: Środowisko gry
        policy1: Polityka agenta 1
        policy2: Polityka agenta 2
        max_steps: Maksymalna liczba kroków
    """
    state = env.reset()
    
    # Początkowy stan
    render_game_state(env, 0)
    input("\nNaciśnij ENTER aby wykonać pierwszy krok...")
    
    total_reward_1 = 0
    total_reward_2 = 0
    
    for step in range(1, max_steps + 1):
        # Pobierz stan z perspektywy każdego agenta
        agent1_state = env.get_agent_state(state, '1')
        agent2_state = env.get_agent_state(state, '2')
        
        # Agent 1 wybiera akcję według polityki
        if agent1_state in policy1:
            action1 = policy1[agent1_state]
        else:
            # Jeśli stan nie jest w polityce, wybierz losową akcję
            actions1 = env.get_possible_actions(agent1_state, '1')
            action1 = random.choice(actions1) if actions1 else 0
            print(f"  [INFO] Agent 1: Stan nieznany, wybrano losową akcję")
        
        # Agent 2 wybiera akcję według polityki
        if agent2_state in policy2:
            action2 = policy2[agent2_state]
        else:
            # Jeśli stan nie jest w polityce, wybierz losową akcję
            actions2 = env.get_possible_actions(agent2_state, '2')
            action2 = random.choice(actions2) if actions2 else 0
            print(f"  [INFO] Agent 2: Stan nieznany, wybrano losową akcję")
        
        # Wykonaj krok
        state, rewards, done, info = env.step(action1, action2)
        total_reward_1 += rewards['1']
        total_reward_2 += rewards['2']
        
        # Renderuj stan po ruchu
        render_game_state(env, step, action1, action2, rewards)
        
        # Wyświetl specjalne zdarzenia
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
        
        input("\nNaciśnij ENTER aby wykonać kolejny krok...")
    
    if not done:
        print("\n" + "="*70)
        print("OSIĄGNIĘTO MAKSYMALNĄ LICZBĘ KROKÓW")
        print("="*70)
        print(f"\nKońcowe punkty:")
        print(f"  Agent 1: {env.agent_score['1']}")
        print(f"  Agent 2: {env.agent_score['2']}")


if __name__ == "__main__":

    env = MTH(MAP)
    env.reset()
    v1, p1 = get_agents_states(env, '1')
    v2, p2 = get_agents_states(env, '2')
    # print(len(v1))
    # print(len(v2))
    # print(len(p1))
    # print(len(p2))
    gamma = 0.9
    theta = 0.001

    policy_agent1, v_agent1 = value_iteration(env, gamma, theta, '1')
    policy_agent2, v_agent2 = value_iteration(env, gamma, theta, '2')

    # Uruchom grę testową z wizualizacją
    print("\n" + "="*70)
    print("ROZPOCZYNAM GRĘ TESTOWĄ Z WYUCZONYMI POLITYKAMI")
    print("="*70)
    play_game_step_by_step(env, policy_agent1, policy_agent2, max_steps=100)
    