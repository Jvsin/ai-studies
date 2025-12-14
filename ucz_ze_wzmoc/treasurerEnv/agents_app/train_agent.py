from treasure_env import MultiTreasureHunterMDP
from linear_agent import LinearApproxAgent
from game_display import play_game_step_by_step
import numpy as np

EPISODES = 1000


# MAP = [
#     "A....H...T",
#     ".#.#...##.",
#     ".H.#.T.#H.",
#     ".##...H.#.",
#     "T...H....B"
# ]

MAP = [
    "A...T#",
    "#....H",
    "#.#T.#",
    "H....#",
    "#T...B",
]

def train_agents(episodes=1000):
    env = MultiTreasureHunterMDP(MAP)
    
    agent1 = LinearApproxAgent(env, '1', alpha=0.01, epsilon=0.2)
    agent2 = LinearApproxAgent(env, '2', alpha=0.01, epsilon=0.2)
    
    print(f"Rozpoczynam trening przez {episodes} epizodów...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        decay = max(0.01, 0.2 * (1 - ep / episodes))
        agent1.epsilon = decay
        agent2.epsilon = decay
        
        while not done and steps < 200:
            act1 = agent1.choose_action(state)
            act2 = agent2.choose_action(state)
            
            next_state, rewards, done, _ = env.step(act1, act2)
            
            agent1.learn(state, act1, rewards['1'], next_state, done)
            agent2.learn(state, act2, rewards['2'], next_state, done)
            
            state = next_state
            steps += 1
            
        if ep % 100 == 0:
            # print(f"Epizod {ep}: Wagi Agenta 1: {np.round(agent1.weights, 2)}")
            print(f"Epizod {ep}: Wagi Agenta 2: {np.round(agent2.weights, 2)}")

    return agent1, agent2, env

if __name__ == "__main__":
    a1, a2, env = train_agents(episodes=EPISODES)
    
    print("\n" + "="*50)
    print("TRENING ZAKOŃCZONY")
    print("Ostateczne wagi Agenta 1 (Interpretacja):")
    w = a1.weights
    print(f"  [0] Bias (Stała):            {w[0]:.2f}")
    print(f"  [1] Ściana/Poza mapą (unikać): {w[1]:.2f} (Powinno być mocno ujemne)")
    print(f"  [2] Dziura (unikać):         {w[2]:.2f} (Powinno być mocno ujemne)")
    print(f"  [3] Dystans (minimalizować): {w[3]:.2f} (Powinno być ujemne)")
    print(f"  [4] Podniesienie (nagroda):  {w[4]:.2f} (Powinno być dodatnie)")
    print(f"  [5] Odniesienie (nagroda):   {w[5]:.2f} (Powinno być bardzo dodatnie)")
    print(f"  [6] P(LEFT):                 {w[6]:.2f}")
    print(f"  [7] P(DOWN):                 {w[7]:.2f}")
    print(f"  [8] P(RIGHT):                {w[8]:.2f}")
    print(f"  [9] P(UP):                   {w[9]:.2f}")
    print("="*50 + "\n")

    # 2. Wyłączenie eksploracji do pokazu (chcemy, żeby grały najlepiej jak umieją)
    a1.epsilon = 0.0
    a2.epsilon = 0.0
    
    # 3. Opakowanie agentów w "polityki" dla game_display
    # game_display oczekuje słownika lub obiektu z __getitem__, 
    # więc robimy prosty wrapper
    class LinearPolicyWrapper:
        def __init__(self, agent):
            self.agent = agent
        def __contains__(self, state):
            return True
        def __getitem__(self, state):
            return self.agent.choose_action(state)
            
    p1 = LinearPolicyWrapper(a1)
    p2 = LinearPolicyWrapper(a2)
    
    # 4. Gra
    play_game_step_by_step(env, p1, p2, max_steps=100)