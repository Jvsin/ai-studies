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
        self.winning_score = len([tile for row in self.original_map for tile in row if tile == 'T']) // 2 + 1

    def reset(self):
        self.map = [row[:] for row in self.original_map]
        self.treasures = set()
        self.bases = {}
        self.agent_holding = {'1': False, '2': False}
        self.agent_score = {'1': 0, '2': 0}

        for y in range(self.height):
            for x in range(self.width):
                tile = self.map[y][x]
                if tile == 'T':
                    self.treasures.add((x, y))
                elif tile == 'A':
                    self.bases['1'] = (x, y)
                    self.map[y][x] = '.'
                elif tile == 'B':
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
    
    def get_agent_state(self, state, agent_id):
        pos1, pos2, treasures, hold1, hold2 = state
        
        if agent_id == '1':
            return (pos1, pos2, frozenset(treasures), hold1, hold2)
        else:
            return (pos2, pos1, frozenset(treasures), hold2, hold1)


    def get_possible_actions(self, state=None, agent='1'):
        if state is None:
            return [LEFT, DOWN, RIGHT, UP]
        
        my_pos, _, _, _, _ = state
        x, y = my_pos

        # pos1, pos2, _, _, _ = state
        # agent_pos = pos1 if agent_id == '1' else pos2
        # x, y = agent_pos
        
        possible_actions = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            _x, _y = ACTIONS[action]
            new_x, new_y = x + _x, y + _y
            
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if self.map[new_y][new_x] != '#':
                    possible_actions.append(action)
        
        return possible_actions
    
    def get_opponent_actions(self, agent_id, state=None):
        pos1, pos2, _, _, _ = state
        op_pos = pos2 if agent_id == '2' else pos1
        x, y = op_pos

        possible_actions = []
        for action in [LEFT, DOWN, RIGHT, UP]:
            _x, _y = ACTIONS[action]
            new_x, new_y = x + _x, y + _y
            
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                if self.map[new_y][new_x] != '#':
                    possible_actions.append(action)
        
        return possible_actions

    
    def get_next_states(self, current_state, action, agent_id='1'):
        my_pos, op_pos, treasures, my_hold, op_hold = current_state
        x, y = my_pos
        _x, _y = ACTIONS[action]
        new_x, new_y = x + _x, y + _y
        
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            return [current_state]  
        if self.map[new_y][new_x] == '#':
            return [current_state]  
        
        new_pos = (new_x, new_y)
        new_op_pos = op_pos
        new_treasures = set(treasures)
        new_my_hold = my_hold
        new_op_hold = op_hold
        my_base = self.bases[agent_id]
        
        if my_pos == op_pos:
            new_pos = self.bases[agent_id]
            new_op_pos = self.bases['2' if agent_id == '1' else '2']
        
            if my_hold:
                new_my_hold = False
                new_treasures.add(my_pos)
            if op_hold:
                new_op_hold = False
                new_treasures.add(op_pos)
            # return [(new_pos, new_op_pos, frozenset(new_treasures), new_my_hold, new_op_hold)]

        if self.map[new_y][new_x] == 'H':
            new_pos = my_base
            if my_hold:
                new_treasures.add(my_pos)
                new_my_hold = False
        else:
            if new_pos in new_treasures and not my_hold:
                new_treasures.remove(new_pos)
                new_my_hold = True
            
            if my_hold and new_pos == my_base:
                new_my_hold = False
        
        ## sprawdzać potencjalne akcje opponenta
        # op_next_actions = self.get_opponent_actions('2' if agent_id == '1' else '1', state=current_state)
        # next_states = []
        # for op_action in op_next_actions:
        #     new_op_pos = op_pos + ACTIONS[op_action]
        #     next_state = (
        #         new_pos,
        #         new_op_pos,
        #         frozenset(new_treasures),
        #         new_my_hold,
        #         new_op_hold
        #     )
        #     next_states.append(next_state)
        
        # return next_states

        
        next_state = (
            new_pos,
            new_op_pos,
            frozenset(new_treasures),
            new_my_hold,
            new_op_hold
        )
        
        return [next_state]
        

    # def get_reward(self, current_state, action, next_state, agent_id='1'):
    #     my_pos_old, _, treasures_old, my_hold_old, _ = current_state
    #     my_pos_new, op_pos_new, treasures_new, my_hold_new, _ = next_state
    #     x, y = my_pos_new
    #     my_base = self.bases[agent_id]

    #     reward = -1
        
    #     if my_pos_new == op_pos_new:
    #         reward -= 5
        
    #     if self.map[y][x] == 'H':
    #         reward -= 30
        
    #     # Podniesienie skarbu (skarb był w treasures_old, ale nie jest w treasures_new)
    #     if my_pos_new in treasures_old and my_pos_new not in treasures_new and not my_hold_old and my_hold_new:
    #         reward += 4
        
    #     # Oddanie skarbu do swojej bazy
    #     if my_hold_old and not my_hold_new and my_pos_new == my_base:
    #         reward += 8

    #     return reward
    
    def get_reward(self, current_state, action, next_state, agent):
        # my_pos, _, treasures, _, _ = current_state
        # x, y = my_pos
        # _x, _y = ACTIONS[action]

        my_pos, op_pos, treasures, my_hold, op_hold = next_state
        x, y = my_pos

        reward = -1
        if my_pos == op_pos:
            reward += -5
        if self.map[y][x] == 'H':
            reward += -30
        if my_pos in self.treasures and not my_hold:
            reward += 4
        if my_pos == self.bases[agent] and my_hold:
            reward += 8

        return reward


    def _move(self, agent_id, action):
        x, y = self.agent_pos[agent_id]
        _x, _y = ACTIONS[action]
        new_x, new_y = x + _x, y + _y

        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            return x, y
        if self.map[new_y][new_x] == '#':
            return x, y
        
        other_pos = self.agent_pos['2'] if agent_id == '1' else self.agent_pos['1']
        if (new_x, new_y) == other_pos:
            return x, y

        return new_x, new_y

    def step(self, action1, action2):
        rewards = {'1': 0.0, '2': 0.0}
        info = {}

        prev_pos1 = self.agent_pos['1']
        prev_pos2 = self.agent_pos['2']

        self.agent_pos['1'] = self._move('1', action1)
        self.agent_pos['2'] = self._move('2', action2)
        rewards['1'] -= 1
        rewards['2'] -= 1

        if self.agent_pos['1'] == self.agent_pos['2']:
            collision_pos = self.agent_pos['1']
            
            if self.agent_holding['1']:
                self.treasures.add(prev_pos1)
                self.agent_holding['1'] = False
                info['1_drop_collision'] = True
            
            if self.agent_holding['2']:
                self.treasures.add(prev_pos2)
                self.agent_holding['2'] = False
                info['2_drop_collision'] = True
            
            self.agent_pos['1'] = self.bases['1']
            self.agent_pos['2'] = self.bases['2']
            
            rewards['1'] -= 5
            rewards['2'] -= 5
            info['collision'] = True

        for agent_id, pos, prev_pos in [('1', self.agent_pos['1'], prev_pos1), ('2', self.agent_pos['2'], prev_pos2)]:
            x, y = pos

            if self.map[y][x] == 'H':
                if self.agent_holding[agent_id]:
                    self.treasures.add(prev_pos)
                    self.agent_holding[agent_id] = False
                    info[f'{agent_id}_drop_trap'] = True
                
                self.agent_pos[agent_id] = self.bases[agent_id]
                rewards[agent_id] = -30
                info[f'{agent_id}_trap'] = True
                continue

            if pos in self.treasures and not self.agent_holding[agent_id]:
                self.treasures.remove(pos)
                self.agent_holding[agent_id] = True
                rewards[agent_id] += 4
                info[f'{agent_id}_pick'] = True

            if pos == self.bases[agent_id] and self.agent_holding[agent_id]:
                self.agent_holding[agent_id] = False
                self.agent_score[agent_id] += 1  
                rewards[agent_id] += 8 
                info[f'{agent_id}_deposit'] = True

        done = self._is_done()
        return self._get_state(), rewards, done, info

    def _is_done(self):
        # check_trap1 = self.map[self.agent_pos['1'][1]][self.agent_pos['1'][0]] == 'H'
        # check_trap2 = self.map[self.agent_pos['2'][1]][self.agent_pos['2'][0]] == 'H'

        if self.agent_score['1'] == self.winning_score or self.agent_score['2'] == self.winning_score:
            return True

        check_treasure = len(self.treasures) == 0

        check_both_home = (
            self.agent_pos['1'] == self.bases['1'] and
            self.agent_pos['2'] == self.bases['2'] and
            not self.agent_holding['1'] and not self.agent_holding['2']
        )
        # return (check_trap1 and check_trap2) or (check_treasure and check_both_home)
        return check_treasure and check_both_home

    def get_all_states(self):
        all_states = []
        
        # Znajdź wszystkie możliwe pozycje dla agentów (bez ścian)
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.map[y][x] != '#':
                    valid_positions.append((x, y))
        
        # Znajdź wszystkie możliwe pozycje skarbów (oryginalne + puste pola)
        treasure_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.original_map[y][x] in ['T', '.', 'A', 'B']:
                    treasure_positions.append((x, y))
        
        # Generuj wszystkie możliwe kombinacje skarbów (podzbiory)
        # Dla uproszczenia, rozważamy tylko pierwotne pozycje skarbów
        initial_treasures = []
        for y in range(self.height):
            for x in range(self.width):
                if self.original_map[y][x] == 'T':
                    initial_treasures.append((x, y))
        
        # Generuj wszystkie podzbiory skarbów (2^n możliwości)
        from itertools import combinations
        treasure_subsets = []
        for r in range(len(initial_treasures) + 1):
            for subset in combinations(initial_treasures, r):
                treasure_subsets.append(frozenset(subset))
        
        # Generuj wszystkie stany
        for pos1 in valid_positions:
            for pos2 in valid_positions:
                for treasures in treasure_subsets:
                    for hold1 in [False, True]:
                        for hold2 in [False, True]:
                            state = (pos1, pos2, treasures, hold1, hold2)
                            all_states.append(state)
        
        return all_states

    def render(self):
        pass
        # grid = [row[:] for row in self.map]
        # for x, y in self.treasures:
        #     grid[y][x] = 'T'
        # p1, p2 = self.agent_pos['1'], self.agent_pos['2']
        # grid[p1[1]][p1[0]] = if self.agent_holding['1'] else '1'
        # grid[p2[1]][p2[0]] = if self.agent_holding['2'] else '2'

        # print("\n" + "="*40)
        # for row in grid:
        #     print(''.join(row))
        # print(f"Skarby: {len(self.treasures)} | Punkty: 1:{self.agent_score['1']}  2:{self.agent_score['2']}")
        # print(f"Trzyma: 1:{self.agent_holding['1']}  2:{self.agent_holding['2']}")
        # print("="*40)