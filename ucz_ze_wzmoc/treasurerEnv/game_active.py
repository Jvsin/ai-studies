import os
import time
import random

try:
    import dill as pickle
except ImportError:
    import pickle

from treasure_env import MultiTreasureHunterMDP, LEFT, DOWN, RIGHT, UP
from agents import QLearningAgent, DQLearningAgent, SARSAAgent, SARSALambdaAgent, ExpectedSARSAAgent


def play_and_train_multi(env, agent1, agent2, episodes=5000, use_agent_perspective=True):
    rewards_history_agent1 = []
    rewards_history_agent2 = []
    scores_history_agent1 = []
    scores_history_agent2 = []

    for ep in range(episodes):
        state = env.reset()
        total_r1 = total_r2 = 0
        agent1.reset()
        agent2.reset()

        if use_agent_perspective:
            state1 = env.get_agent_state(state, '1')
            state2 = env.get_agent_state(state, '2')
        else:
            state1 = state2 = state
        
        action1 = agent1.get_action(state1)
        action2 = agent2.get_action(state2)

        while True:
            next_state, rewards, done, _ = env.step(action1, action2)
            r1, r2 = rewards['1'], rewards['2']
            total_r1 += r1
            total_r2 += r2

            if use_agent_perspective:
                next_state1 = env.get_agent_state(next_state, '1')
                next_state2 = env.get_agent_state(next_state, '2')
            else:
                next_state1 = next_state2 = next_state

            next_action1 = agent1.update(state1, action1, r1, next_state1)
            next_action2 = agent2.update(state2, action2, r2, next_state2)

            action1 = next_action1 if next_action1 is not None else agent1.get_action(next_state1)
            action2 = next_action2 if next_action2 is not None else agent2.get_action(next_state2)
            
            state1 = next_state1
            state2 = next_state2
            state = next_state

            if done:
                break

        score1 = env.agent_score['1']
        score2 = env.agent_score['2']
        
        rewards_history_agent1.append(total_r1)
        rewards_history_agent2.append(total_r2)
        scores_history_agent1.append(score1)
        scores_history_agent2.append(score2)

        if (ep + 1) % 100 == 0:
            print(f"Epizod {ep+1:4d} | Agent1: R={total_r1:6.1f} P={score1} | Agent2: R={total_r2:6.1f} P={score2}")

    return (rewards_history_agent1, rewards_history_agent2, 
            scores_history_agent1, scores_history_agent2)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_game(env, step, last_actions=None, last_rewards=None, last_info=None):
    grid = [row[:] for row in env.map]
    
    for x, y in env.treasures:
        grid[y][x] = 'T'
    
    p1, p2 = env.agent_pos['1'], env.agent_pos['2']
    
    if p1 == p2:
        grid[p1[1]][p1[0]] = 'X'
    else:
        grid[p1[1]][p1[0]] = '*' if env.agent_holding['1'] else '1'
        grid[p2[1]][p2[0]] = '+' if env.agent_holding['2'] else '2'
    
    print("=" * 60)
    print(f"  TREASURE HUNTER - Krok {step}".center(60))
    print("=" * 60)
    
    print("\n  " + "".join(str(i) for i in range(len(grid[0]))))
    for idx, row in enumerate(grid):
        print(f"{idx} " + "".join(row))
    
    print("\nLegenda:")
    print("  A/B = Bazy agentow | T = Skarb | H = Pulapka | # = Sciana")
    print("  1/2 = Agenci | */+ = Agent trzyma skarb | X = Kolizja")
    
    print("\nStatus:")
    print(f"  Punkty:  Agent 1: {env.agent_score['1']}  |  Agent 2: {env.agent_score['2']}")
    print(f"  Skarby:  Na mapie: {len(env.treasures)}  |  A1 trzyma: {env.agent_holding['1']}  |  A2 trzyma: {env.agent_holding['2']}")
    
    if last_actions and last_rewards:
        action_names = {0: '< LEWO', 1: 'v DOL', 2: '> PRAWO', 3: '^ GORA'}
        print(f"\nOstatni ruch:")
        print(f"  Agent 1: {action_names[last_actions[0]]} -> Nagroda: {last_rewards['1']:+.0f}")
        print(f"  Agent 2: {action_names[last_actions[1]]} -> Nagroda: {last_rewards['2']:+.0f}")
        
        if last_info:
            events = []
            if '1_pick' in last_info:
                events.append("[PICK] Agent 1 podniosl skarb!")
            if '2_pick' in last_info:
                events.append("[PICK] Agent 2 podniosl skarb!")
            if '1_deposit' in last_info:
                events.append("[OK] Agent 1 oddal skarb do bazy!")
            if '2_deposit' in last_info:
                events.append("[OK] Agent 2 oddal skarb do bazy!")
            
            if events:
                print(f"  ** {' | '.join(events)}")
    
    print("=" * 60)

def test_policy_animated(env, agent1, agent2, episodes=3, delay=1.0, max_steps=100, use_agent_perspective=True):
    agent1.turn_off_learning()
    agent2.turn_off_learning()
    
    for ep in range(episodes):
        state = env.reset()
        step = 0
        
        clear_console()
        print(f"\nGRA {ep+1}/{episodes} - START!\n")
        render_game(env, step)
        time.sleep(delay * 1.5)  # Dłuższa pauza na początku
        
        while step < max_steps:
            if use_agent_perspective:
                state1 = env.get_agent_state(state, '1')
                state2 = env.get_agent_state(state, '2')
            else:
                state1 = state2 = state
            
            a1 = agent1.get_action(state1)
            a2 = agent2.get_action(state2)
            
            state, rewards, done, info = env.step(a1, a2)
            step += 1
            
            clear_console()
            render_game(env, step, last_actions=(a1, a2), last_rewards=rewards, last_info=info)
            
            if done:
                time.sleep(delay)
                print("\n" + "=" * 60)
                print(f"KONIEC GRY {ep+1} (w {step} krokach)".center(60))
                print("=" * 60)
                
                score1 = env.agent_score['1']
                score2 = env.agent_score['2']
                
                print(f"\nWYNIK KONCOWY:")
                print(f"  Agent 1: {score1} punktow")
                print(f"  Agent 2: {score2} punktow")
                
                if score1 > score2:
                    print(f"\n*** WYGRYWA Agent 1 z przewaga {score1 - score2} punktow!")
                elif score2 > score1:
                    print(f"\n*** WYGRYWA Agent 2 z przewaga {score2 - score1} punktow!")
                else:
                    print(f"\nREMIS! Obaj agenci zdobyli {score1} punktow!")
                
                print("=" * 60)
                
                if ep < episodes - 1:
                    input("\n[PAUZA] Nacisnij ENTER aby rozpoczac nastepna gre...")
                break
            
            time.sleep(delay)
        
        if step >= max_steps:
            print(f"\n[!] PRZERWANO - osiagnieto limit {max_steps} krokow!")
            print(f"Punkty: A1={env.agent_score['1']}, A2={env.agent_score['2']}")
            
            if ep < episodes - 1:
                input("\nNacisnij ENTER aby rozpoczac nastepna gre...")

def load_agents(agents_path):
    with open(agents_path, 'rb') as f:
        agents_dict = pickle.load(f)
    
    if isinstance(agents_dict, dict) and 'agent1' in agents_dict and 'agent2' in agents_dict:
        print(f"[>] Wczytano: {agents_dict.get('description', 'Para agentów')}")
        return agents_dict['agent1'], agents_dict['agent2']
    else:
        raise ValueError("Plik nie zawiera pary agentów w oczekiwanym formacie!")

if __name__ == "__main__":
    MAP2 = [
        "A........T",
        "..#..#..#.",
        ".H..T...H.",
        ".#..#..#..",
        "T........B"
    ]
    MAP3 = [
        "A....H...T",
        ".#.#....#.",
        ".H.#T..#H.",
        "..#...H.#.",
        "T...H....B"
    ]
    
    print("=" * 60)
    print("TREASURE HUNTER - Test Wytrenowanych Agentow".center(60))
    print("=" * 60)
    print("\nWybierz tryb:")
    print("1. Zaladuj wytrenowanych agentow z plikow .pkl")
    print("2. Stworz nowych agentow (bez treningu - losowe ruchy)")
    
    choice = input("\nWybor (1/2): ").strip()
    
    env = MultiTreasureHunterMDP(MAP3)
    
    if choice == "1":
        current_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
        pkl_files = [f for f in os.listdir(current_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            print("\nERROR: Nie znaleziono zadnych plikow .pkl w folderze!")
            print(f"Folder: {current_dir}")
            print("\nUruchom najpierw trening w notebooku i zapisz agentow!")
            exit(1)
        
        print(f"\nZnaleziono {len(pkl_files)} plikow .pkl:")
        for idx, fname in enumerate(pkl_files, 1):
            print(f"  {idx}. {fname}")
        
        print("\nWybierz plik z parą agentów:")
        while True:
            try:
                idx = int(input(f"Numer (1-{len(pkl_files)}): "))
                if 1 <= idx <= len(pkl_files):
                    agents_file = pkl_files[idx - 1]
                    break
                print(f"Podaj liczbe od 1 do {len(pkl_files)}")
            except ValueError:
                print("Podaj poprawny numer!")
        
        try:
            agents_path = os.path.join(current_dir, agents_file)
            
            print(f"\n[>] Ladowanie agentow z: {agents_file}")
            
            agent1, agent2 = load_agents(agents_path)
            
            agent1.get_legal_actions = lambda s: env.get_possible_actions(s, agent_id='1')
            agent2.get_legal_actions = lambda s: env.get_possible_actions(s, agent_id='2')
            
            print("[OK] Agenci zaladowani. Rozpoczynam grę...")
            
        except Exception as e:
            print(f"\nBlad podczas ladowania agentow: {e}")
            print("Sprawdz czy pliki sa poprawne i zostaly zapisane przez dill/pickle")
            exit(1)
    else:
        
        print("\nTworzenie nowych agentow...")
        agent1 = SARSALambdaAgent(
            alpha=0.1, epsilon=0.25, discount=0.99,
            get_legal_actions=lambda s: env.get_possible_actions(s, agent_id='1'),
            lambda_value=0.9
        )
        agent2 = DQLearningAgent(
            alpha=0.1, epsilon=0.25, discount=0.99,
            get_legal_actions=lambda s: env.get_possible_actions(s, agent_id='2')
        )
        print("Agenci utworzeni (niewtrenowani - beda robic losowe ruchy)")
    
    print("\nKonfiguracja:")
    episodes = int(input("Liczba gier do rozegrania (domyslnie 3): ") or "3")
    delay = float(input("Opoznienie miedzy ruchami w sekundach (domyslnie 0.5): ") or "0.5")
    
    input("\nNacisnij ENTER aby rozpoczac...")
    
    test_policy_animated(env, agent1, agent2, episodes=episodes, delay=delay, max_steps=100)
    
    print("\n\nDziekuje za gre!")
