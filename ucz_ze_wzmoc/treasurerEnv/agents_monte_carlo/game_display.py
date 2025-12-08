import random

def render_game_state(env, step_num, state, action1=None, action2=None, rewards=None):
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
    print(f"Stan: {state}")
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
    render_game_state(env, state, 0)
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
        render_game_state(env, step, state, action1, action2, rewards)
        
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