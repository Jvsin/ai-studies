from treasure_env import MultiTreasureHunterMDP as MTH, ACTIONS, LEFT, DOWN, RIGHT, UP
from game_display import play_game_step_by_step
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

MAP = [
    "A....H...T",
    ".#.#...##.",
    ".H.#.T.#H.",
    ".##...H.#.",
    "T...H....B"
]

def get_action_epsilon_greedy(Q, state, possible_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(possible_actions)
    
    if state not in Q:
        return random.choice(possible_actions)
    
    best_value = -float('inf')
    
    for a in possible_actions:
        val = Q[state].get(a, 0.0)
        if val > best_value:
            best_value = val

    best_actions = [a for a in possible_actions if Q[state].get(a, 0.0) == best_value]
    
    return random.choice(best_actions)

def generate_episode(env, agent_id, Q, epsilon, opponent_policy=None):
    state = env.reset()
    episode = []
    done = False
    steps = 0
    max_steps = 200
    
    opponent_id = '2' if agent_id == '1' else '1'

    while not done and steps < max_steps:
        possible_actions = env.get_possible_actions(state, agent_id)
        
        action = get_action_epsilon_greedy(Q, state, possible_actions, epsilon)
        
        op_actions = env.get_possible_actions(state, opponent_id)
        if opponent_policy and state in opponent_policy:
            op_action = opponent_policy[state]
        else:
            op_action = random.choice(op_actions)

        if agent_id == '1':
            act1, act2 = action, op_action
        else:
            act1, act2 = op_action, action

        next_state, rewards, done, _ = env.step(act1, act2)
        reward = rewards[agent_id]
        
        episode.append((state, action, reward))
        
        state = next_state
        steps += 1
        
    return episode

def monte_carlo_first_visit(env, agent_id, num_episodes, gamma=0.9, epsilon=0.1, opponent_policy=None):
    Q = {} 
    Returns = {} 

    for i in range(1, num_episodes + 1):
        
        current_epsilon = max(0.01, epsilon * (1 - i / num_episodes))
        episode = generate_episode(env, agent_id, Q, current_epsilon, opponent_policy)
        
        G = 0
        visited_pairs_in_episode = set()
        
        for t in range(len(episode) - 1, -1, -1):
            st, at, r_t1 = episode[t]
            
            G = gamma * G + r_t1
            
            # Sprawdzenie First-Visit:
            # Musimy sprawdzić, czy para (st, at) wystąpiła WRAZEŚNIEJ w tym samym epizodzie
            # Najprostszy sposób w pętli od tyłu: sprawdzamy czy (st, at) występuje w części listy episode[0:t]
            # Ale dla wydajności używa się setu w pętli "od przodu" lub założenia, 
            # że przy "od tyłu" aktualizujemy, ale nadpisujemy jeśli wystąpi wcześniej.
            # Zgodnie z Twoim pseudokodem: "Jeżeli para st,at nie pojawiła się wcześniej"
            
            # Metoda Pythoniczna: sprawdź czy (st, at) jest w slice episode[:t]
            # To jest kosztowne obliczeniowo O(N^2), ale zgodne z algorytmem.
            # Optymalizacja: sprawdzamy to tylko logicznie.
            
            has_appeared_before = False
            for k in range(t):
                if episode[k][0] == st and episode[k][1] == at:
                    has_appeared_before = True
                    break
            
            if not has_appeared_before:
                if st not in Returns:
                    Returns[st] = {}
                    Q[st] = {}
                if at not in Returns[st]:
                    Returns[st][at] = []
                    Q[st][at] = 0

                Returns[st][at].append(G)
                
                all_returns = Returns[st][at]
                Q[st][at] = sum(all_returns) / len(all_returns)
                
                # a* = argmax(Q(st, a)) oraz aktualizacja pi (epsilon-greedy)
                # Dzieje się "automatycznie" w funkcji get_action_epsilon_greedy,
                # która zawsze patrzy na zaktualizowane Q.
        
        if i % 1000 == 0:
            print(f"Epizod {i}/{num_episodes} | Epsilon: {current_epsilon:.3f}")

    final_policy = {}
    for s in Q:
        if Q[s]:
            best_a = max(Q[s], key=Q[s].get)
            final_policy[s] = best_a
            
    return final_policy, Q

if __name__ == "__main__":
    env = MTH(MAP)
    
    EPISODES = 10000 
    GAMMA = 0.95
    EPSILON = 0.5

    print("Trening agenta 1:")
    policy_agent1, Q1 = monte_carlo_first_visit(env, '1', EPISODES, GAMMA, EPSILON)
    print(len(Q1))

    print("Trening agenta 2:")
    policy_agent2, Q2 = monte_carlo_first_visit(env, '2', EPISODES, GAMMA, EPSILON)
    print(len(Q2))

    env.reset()
    play_game_step_by_step(env, policy_agent1, policy_agent2, max_steps=100)