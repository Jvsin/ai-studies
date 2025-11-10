"""
Środowisko gry Treasure Hunter dla wielu agentów
"""

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

class MultiTreasureHunterMDP:
    def __init__(self, map_lines):
        self.original_map = [list(row) for row in map_lines]
        self.height = len(self.original_map)
        self.width = len(self.original_map[0])
        self.treasures = set()
        self.bases = {}
        self.agent_pos = {}
        self.agent_holding = {'1': False, '2': False}
        self.agent_score = {'1': 0, '2': 0}

    def reset(self):
        self.map = [row[:] for row in self.original_map]
        self.treasures = set()
        self.bases = {}
        self.agent_holding = {'1': False, '2': False}
        self.agent_score = {'1': 0, '2': 0}

        # Najpierw znajdź skarby i bazy
        for y in range(self.height):
            for x in range(self.width):
                tile = self.map[y][x]
                if tile == 'T':
                    self.treasures.add((x, y))
                elif tile == 'A':
                    # Baza agenta 1
                    self.bases['1'] = (x, y)
                    self.map[y][x] = '.'
                elif tile == 'B':
                    # Baza agenta 2
                    self.bases['2'] = (x, y)
                    self.map[y][x] = '.'

        self.agent_pos['1'] = self.bases['1']
        self.agent_pos['2'] = self.bases['2']

        return self._get_state()

    def _get_state(self):
        return (
            self.agent_pos['1'],
            self.agent_pos['2'],
            frozenset(self.treasures),
            self.agent_holding['1'],
            self.agent_holding['2']
        )

    def get_possible_actions(self, state=None):
        """Zwraca faktycznie możliwe akcje dla obecnego stanu"""
        if state is None:
            # Jeśli nie podano stanu, zwróć wszystkie akcje (dla inicjalizacji)
            return [LEFT, DOWN, RIGHT, UP]
        
        # Rozpakuj stan
        pos1, pos2, treasures, holding1, holding2 = state
        
        # Sprawdź możliwe akcje dla obu agentów (unikalny zbiór)
        possible = set()
        
        for agent_pos in [pos1, pos2]:
            x, y = agent_pos
            
            # Sprawdź każdą akcję
            for action in [LEFT, DOWN, RIGHT, UP]:
                dx, dy = ACTIONS[action]
                nx, ny = x + dx, y + dy
                
                # Sprawdź czy nie wychodzi poza mapę
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Sprawdź czy nie jest ścianą
                    if self.map[ny][nx] != '#':
                        possible.add(action)
        
        # Jeśli żadna akcja nie jest możliwa (teoretycznie nie powinno się zdarzyć)
        # zwróć wszystkie akcje jako fallback
        return list(possible) if possible else [LEFT, DOWN, RIGHT, UP]

    def _move(self, agent_id, action):
        x, y = self.agent_pos[agent_id]
        dx, dy = ACTIONS[action]
        nx, ny = x + dx, y + dy

        # Granice
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return x, y
        # Ściana
        if self.map[ny][nx] == '#':
            return x, y
        # Kolizja z drugim agentem
        other_pos = self.agent_pos['2'] if agent_id == '1' else self.agent_pos['1']
        if (nx, ny) == other_pos:
            return x, y

        return nx, ny

    def step(self, action1, action2):
        rewards = {'1': 0.0, '2': 0.0}
        info = {}

        # Ruch
        self.agent_pos['1'] = self._move('1', action1)
        self.agent_pos['2'] = self._move('2', action2)
        rewards['1'] -= 1
        rewards['2'] -= 1

        # Interakcje
        for aid, pos in [('1', self.agent_pos['1']), ('2', self.agent_pos['2'])]:
            x, y = pos

            # Pułapka
            if self.map[y][x] == 'H':
                rewards[aid] = -30
                continue

            # Zbieranie skarbu
            if pos in self.treasures and not self.agent_holding[aid]:
                self.treasures.remove(pos)
                self.agent_holding[aid] = True
                rewards[aid] += 3
                info[f'{aid}_pick'] = True

            # Odkładanie w bazie
            if pos == self.bases[aid] and self.agent_holding[aid]:
                self.agent_holding[aid] = False
                self.agent_score[aid] += 3
                rewards[aid] += 5
                info[f'{aid}_deposit'] = True

        done = self._is_done()
        return self._get_state(), rewards, done, info

    def _is_done(self):
        check_trap1 = self.map[self.agent_pos['1'][1]][self.agent_pos['1'][0]] == 'H'
        check_trap2 = self.map[self.agent_pos['2'][1]][self.agent_pos['2'][0]] == 'H'

        check_treasure = len(self.treasures) == 0
        both_home = (
            self.agent_pos['1'] == self.bases['1'] and
            self.agent_pos['2'] == self.bases['2'] and
            not self.agent_holding['1'] and not self.agent_holding['2']
        )
        return (check_trap1 and check_trap2) or (check_treasure and both_home)

    def render(self):
        """Pusta metoda - renderowanie jest w test_game.py"""
        pass
