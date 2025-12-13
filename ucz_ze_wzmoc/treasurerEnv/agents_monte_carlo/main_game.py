import time
from treasure_env import MultiTreasureHunterMDP
import game_display
from mcts_agent import mcts_policy

# Mapa gry (Legenda: . = puste, # = ściana, A/B = bazy, T = skarb, H = dziura)
MAP = [
    "#######",
    "#A...T#",
    "#.###.#",
    "#T.H.T#",
    "#.###.#",
    "#B....#",
    "#######"
]

def run_mcts_game():
    # Inicjalizacja środowiska
    env = MultiTreasureHunterMDP(MAP)
    state = env.reset()
    
    max_steps = 100
    step_num = 0
    done = False
    
    total_rewards = {'1': 0, '2': 0}

    # Pierwsze wyświetlenie
    game_display.render_game_state(env, step_num, state)
    print("Rozpoczynanie gry MCTS vs MCTS...")
    time.sleep(1)

    while not done and step_num < max_steps:
        step_num += 1

        action1 = mcts_policy(env, state, agent_id='1')
        
        action2 = mcts_policy(env, state, agent_id='2')

        next_state, rewards, done, info = env.step(action1, action2)
        
        total_rewards['1'] += rewards['1']
        total_rewards['2'] += rewards['2']
        
        state = next_state
        
        game_display.render_game_state(env, step_num, state, action1, action2, rewards)
        
        if info.get('collision'):
            print(">>> KOLIZJA! <<<")
        
        time.sleep(1)

    # Podsumowanie
    print("\n" + "="*30)
    print("KONIEC GRY")
    print(f"Wynik Agenta 1: {env.agent_score['1']} (Nagroda łącznie: {total_rewards['1']})")
    print(f"Wynik Agenta 2: {env.agent_score['2']} (Nagroda łącznie: {total_rewards['2']})")
    
    if env.agent_score['1'] > env.agent_score['2']:
        print("Wygrywa Agent 1!")
    elif env.agent_score['2'] > env.agent_score['1']:
        print("Wygrywa Agent 2!")
    else:
        print("Remis!")

if __name__ == "__main__":
    run_mcts_game()