"""
Skrypt pomocniczy do zapisywania wytrenowanych agentow z notebooka.

Uzycie w notebooku po treningu:
    import pickle
    
    # Zapisz agentow
    with open('treasurerEnv/agent1.pkl', 'wb') as f:
        pickle.dump(agent1, f)
    
    with open('treasurerEnv/agent2.pkl', 'wb') as f:
        pickle.dump(agent2, f)
    
    print("[OK] Agenci zapisani!")

Nastepnie uruchom test_game.py
"""

import pickle
import sys
import os

def save_agents(agent1, agent2, directory='treasurerEnv'):
    """
    Zapisuje agentow do plikow pickle
    
    Args:
        agent1: Pierwszy agent (np. SARSALambdaAgent)
        agent2: Drugi agent (np. DQLearningAgent)
        directory: Folder docelowy (domyslnie 'treasurerEnv')
    """
    # Stworz folder jesli nie istnieje
    os.makedirs(directory, exist_ok=True)
    
    # Zapisz agentow
    agent1_path = os.path.join(directory, 'agent1.pkl')
    agent2_path = os.path.join(directory, 'agent2.pkl')
    
    with open(agent1_path, 'wb') as f:
        pickle.dump(agent1, f)
    print(f"[OK] Agent 1 zapisany do: {agent1_path}")
    
    with open(agent2_path, 'wb') as f:
        pickle.dump(agent2, f)
    print(f"[OK] Agent 2 zapisany do: {agent2_path}")
    
    # Pokaz statystyki
    print(f"\nStatystyki agentow:")
    print(f"  Agent 1: {len(agent1._qvalues)} znanych stanow")
    
    if hasattr(agent2, '_qvaluesA'):
        # Double Q-Learning
        print(f"  Agent 2: {len(agent2._qvaluesA) + len(agent2._qvaluesB)} znanych stanow (Double-Q)")
    else:
        # Zwykly Q-Learning lub SARSA
        print(f"  Agent 2: {len(agent2._qvalues)} znanych stanow")
    
    print(f"\nTeraz mozesz uruchomic: python {os.path.join(directory, 'test_game.py')}")

if __name__ == "__main__":
    print("[!] Ten skrypt jest pomocniczym narzedziem.")
    print("Uzyj go w notebooku po treningu agentow:")
    print()
    print("   import pickle")
    print("   with open('treasurerEnv/agent1.pkl', 'wb') as f:")
    print("       pickle.dump(agent1, f)")
    print("   with open('treasurerEnv/agent2.pkl', 'wb') as f:")
    print("       pickle.dump(agent2, f)")
    print()
    print("Nastepnie uruchom: python treasurerEnv/test_game.py")
