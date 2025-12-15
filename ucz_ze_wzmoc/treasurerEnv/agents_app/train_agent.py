from treasure_env import MultiTreasureHunterMDP
from linear_agent import LinearApproxAgent, AgentWrapper
from game_display import play_game_step_by_step
import numpy as np
import os

TRAIN = True 
EPISODES = 12000
WEIGHTS_FILE_AGENT1 = "agent1_weights.npy"
WEIGHTS_FILE_AGENT2 = "agent2_weights.npy"

MAP = [
    "A........T",
    "..#..#..#.",
    ".H..T...H.",
    ".#..#..#..",
    "T........B"
]

# MAP = [
#     "A...T#",
#     "#....H",
#     "#.#..#",
#     "H..T.#",
#     "#T...B",
# ]

def train_agents(episodes=1000):
    env = MultiTreasureHunterMDP(MAP)
    
    agent1 = LinearApproxAgent(env, '1', alpha=0.01, epsilon=0.2)
    agent2 = LinearApproxAgent(env, '2', alpha=0.01, epsilon=0.2)
    
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
            # print(f"Epizod {ep}: Wagi Agenta 2: {np.round(agent2.weights, 2)}")
            print(f"Epizod {ep}/{EPISODES}")

    return agent1, agent2, env

if __name__ == "__main__":
    env = MultiTreasureHunterMDP(MAP)
    
    if TRAIN:
        a1, a2, _ = train_agents(episodes=EPISODES)
        
        np.save(WEIGHTS_FILE_AGENT1, a1.weights)
        np.save(WEIGHTS_FILE_AGENT2, a2.weights)
    else:
        if not os.path.exists(WEIGHTS_FILE_AGENT1) or not os.path.exists(WEIGHTS_FILE_AGENT2):
            print("File doesn't exist")
            exit(1)
        
        a1 = LinearApproxAgent(env, '1', alpha=0.01, epsilon=0.0)
        a2 = LinearApproxAgent(env, '2', alpha=0.01, epsilon=0.0)
        
        a1.weights = np.load(WEIGHTS_FILE_AGENT1)
        a2.weights = np.load(WEIGHTS_FILE_AGENT2)
        
        print(f"Wczytano wagi Agenta 1 z {WEIGHTS_FILE_AGENT1}")
        print(f"Wczytano wagi Agenta 2 z {WEIGHTS_FILE_AGENT2}")
    
    a1.epsilon = 0.0
    a2.epsilon = 0.0
        
    p1 = AgentWrapper(a1)
    p2 = AgentWrapper(a2)
    
    play_game_step_by_step(env, p1, p2, max_steps=100)