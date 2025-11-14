import os
import time
import random

# Sprobuj uzyc dill (obsługuje lambdy), jesli nie ma to pickle
try:
    import dill as pickle
    print("[OK] Uzywam dill - obsluga lambd")
except ImportError:
    import pickle
    print("[!] Uzywam pickle - jesli blad, zainstaluj: pip install dill")

from treasure_env import MultiTreasureHunterMDP, LEFT, DOWN, RIGHT, UP

def clear_console():
    """Czyści konsolę - działa na Windows i Unix"""
    os.system('cls' if os.name == 'nt' else 'clear')

def render_game(env, step, last_actions=None, last_rewards=None, last_info=None):
    """Renderuje aktualny stan gry w konsoli"""
    # Buduj siatkę
    grid = [row[:] for row in env.map]
    
    # Dodaj skarby
    for x, y in env.treasures:
        grid[y][x] = 'T'
    
    # Dodaj agentów
    p1, p2 = env.agent_pos['1'], env.agent_pos['2']
    
    # Sprawdź kolizję
    if p1 == p2:
        grid[p1[1]][p1[0]] = 'X'  # Kolizja
    else:
        # Agent 1 - pokaż czy trzyma skarb
        grid[p1[1]][p1[0]] = '*' if env.agent_holding['1'] else '1'
        # Agent 2 - pokaż czy trzyma skarb
        grid[p2[1]][p2[0]] = '+' if env.agent_holding['2'] else '2'
    
    # Wyświetl nagłówek
    print("=" * 60)
    print(f"  TREASURE HUNTER - Krok {step}".center(60))
    print("=" * 60)
    
    # Wyświetl planszę
    print("\n  " + "".join(str(i) for i in range(len(grid[0]))))
    for idx, row in enumerate(grid):
        print(f"{idx} " + "".join(row))
    
    # Legenda
    print("\nLegenda:")
    print("  A/B = Bazy agentow | T = Skarb | H = Pulapka | # = Sciana")
    print("  1/2 = Agenci | */+ = Agent trzyma skarb | X = Kolizja")
    
    # Status gry
    print("\nStatus:")
    print(f"  Punkty:  Agent 1: {env.agent_score['1']}  |  Agent 2: {env.agent_score['2']}")
    print(f"  Skarby:  Na mapie: {len(env.treasures)}  |  A1 trzyma: {env.agent_holding['1']}  |  A2 trzyma: {env.agent_holding['2']}")
    
    # Ostatni ruch
    if last_actions and last_rewards:
        action_names = {0: '< LEWO', 1: 'v DOL', 2: '> PRAWO', 3: '^ GORA'}
        print(f"\nOstatni ruch:")
        print(f"  Agent 1: {action_names[last_actions[0]]} -> Nagroda: {last_rewards['1']:+.0f}")
        print(f"  Agent 2: {action_names[last_actions[1]]} -> Nagroda: {last_rewards['2']:+.0f}")
        
        # Eventy specjalne
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

def test_policy_animated(env, agent1, agent2, episodes=3, delay=1.0, max_steps=100):
    """Testuje politykę agentów z animacją w konsoli"""
    agent1.turn_off_learning()
    agent2.turn_off_learning()
    
    for ep in range(episodes):
        state = env.reset()
        step = 0
        
        # Początkowy stan
        clear_console()
        print(f"\nGRA {ep+1}/{episodes} - START!\n")
        render_game(env, step)
        time.sleep(delay * 1.5)  # Dłuższa pauza na początku
        
        while step < max_steps:
            # Pobierz akcje
            a1 = agent1.get_action(state)
            a2 = agent2.get_action(state)
            
            # Wykonaj krok
            state, rewards, done, info = env.step(a1, a2)
            step += 1
            
            # Wyświetl nowy stan
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

def load_agents(agent1_path, agent2_path):
    """Ładuje wytrenowanych agentów z plików pickle"""
    with open(agent1_path, 'rb') as f:
        agent1 = pickle.load(f)
    with open(agent2_path, 'rb') as f:
        agent2 = pickle.load(f)
    return agent1, agent2

if __name__ == "__main__":
    # Definicja mapy
    MAP2 = [
        "A........T",
        "..#..#..#.",
        ".H..T...H.",
        ".#..#..#..",
        "T........B"
    ]
    
    print("=" * 60)
    print("TREASURE HUNTER - Test Wytrenowanych Agentow".center(60))
    print("=" * 60)
    print("\nWybierz tryb:")
    print("1. Zaladuj wytrenowanych agentow z plikow .pkl")
    print("2. Stworz nowych agentow (bez treningu - losowe ruchy)")
    
    choice = input("\nWybor (1/2): ").strip()
    
    # Stwórz środowisko
    env = MultiTreasureHunterMDP(MAP2)
    
    if choice == "1":
        # Znajdź wszystkie pliki .pkl w bieżącym katalogu
        current_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
        pkl_files = [f for f in os.listdir(current_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            print("\n[X] Nie znaleziono zadnych plikow .pkl w folderze!")
            print(f"Folder: {current_dir}")
            print("\nUruchom najpierw trening w notebooku i zapisz agentow!")
            exit(1)
        
        # Wyświetl dostępne pliki
        print(f"\nZnaleziono {len(pkl_files)} plikow .pkl:")
        for idx, fname in enumerate(pkl_files, 1):
            print(f"  {idx}. {fname}")
        
        # Wybór pierwszego agenta
        print("\nWybierz PIERWSZEGO agenta:")
        while True:
            try:
                idx1 = int(input(f"Numer (1-{len(pkl_files)}): "))
                if 1 <= idx1 <= len(pkl_files):
                    agent1_file = pkl_files[idx1 - 1]
                    break
                print(f"Podaj liczbe od 1 do {len(pkl_files)}")
            except ValueError:
                print("Podaj poprawny numer!")
        
        # Wybór drugiego agenta
        print("\nWybierz DRUGIEGO agenta:")
        while True:
            try:
                idx2 = int(input(f"Numer (1-{len(pkl_files)}): "))
                if 1 <= idx2 <= len(pkl_files):
                    agent2_file = pkl_files[idx2 - 1]
                    break
                print(f"Podaj liczbe od 1 do {len(pkl_files)}")
            except ValueError:
                print("Podaj poprawny numer!")
        
        # Ładuj agentów
        try:
            agent1_path = os.path.join(current_dir, agent1_file)
            agent2_path = os.path.join(current_dir, agent2_file)
            
            print(f"\n[>] Ladowanie agentow:")
            print(f"    Agent 1: {agent1_file}")
            print(f"    Agent 2: {agent2_file}")
            
            agent1, agent2 = load_agents(agent1_path, agent2_path)
            
            # WAZNE: Podmien get_legal_actions bo lambda odnosi sie do nieistniejacego env
            agent1.get_legal_actions = lambda s: env.get_possible_actions(s)
            agent2.get_legal_actions = lambda s: env.get_possible_actions(s)
            
            print("[OK] Agenci zaladowani i gotowi do gry!")
            
        except Exception as e:
            print(f"\n[X] Blad podczas ladowania agentow: {e}")
            print("Sprawdz czy pliki sa poprawne i zostaly zapisane przez dill/pickle")
            exit(1)
    else:
        # Importuj z agents.py
        from agents import SARSALambdaAgent, DQLearningAgent
        
        print("\n[>] Tworzenie nowych agentow...")
        agent1 = SARSALambdaAgent(
            alpha=0.1, epsilon=0.25, discount=0.99,
            get_legal_actions=lambda s: env.get_possible_actions(s),
            lambda_value=0.9
        )
        agent2 = DQLearningAgent(
            alpha=0.1, epsilon=0.25, discount=0.99,
            get_legal_actions=lambda s: env.get_possible_actions(s)
        )
        print("[OK] Agenci utworzeni (niewtrenowani - beda robic losowe ruchy)")
    
    # Parametry gry
    print("\nKonfiguracja:")
    episodes = int(input("Liczba gier do rozegrania (domyslnie 3): ") or "3")
    delay = float(input("Opoznienie miedzy ruchami w sekundach (domyslnie 0.5): ") or "0.5")
    
    input("\n[OK] Wszystko gotowe! Nacisnij ENTER aby rozpoczac...")
    
    # Rozpocznij test
    test_policy_animated(env, agent1, agent2, episodes=episodes, delay=delay, max_steps=100)
    
    print("\n\nDziekuje za gre!")
