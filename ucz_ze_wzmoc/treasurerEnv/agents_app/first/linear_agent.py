import numpy as np
import random
from treasure_env import ACTIONS, LEFT, DOWN, RIGHT, UP

class LinearApproxAgent:
    def __init__(self, env, agent_id='1', alpha=0.01, gamma=0.9, epsilon=0.1):
        self.env = env
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon   
        
        self.weights = np.zeros(10) # bias + cechy + prawdopodobieństwa akcji
        
    def extract_features(self, state, action):
        pos1, pos2, treasures, hold1, hold2 = state
        
        if self.agent_id == '1':
            my_pos = pos1
            op_pos = pos2
            my_hold = hold1
            # op_hold = hold2
        else:
            my_pos = pos2
            op_pos = pos1
            my_hold = hold2
            # op_hold = hold1
            
        dx, dy = ACTIONS[action]
        next_x = my_pos[0] + dx
        next_y = my_pos[1] + dy
        
        features = np.zeros(10)
        
        features[0] = 1.0
        
        if not (0 <= next_x < self.env.width and 0 <= next_y < self.env.height):
            features[1] = 1.0 
            return features
        
        # czy ściana
        if self.env.map[next_y][next_x] == '#':
            features[1] = 1.0 
            return features

        # czy dziura
        if self.env.map[next_y][next_x] == 'H':
            features[2] = 1.0
            
        target_pos = None
        base_pos = self.env.bases[self.agent_id]
        
        if my_hold:
            target_pos = base_pos
        elif treasures:
            dists = [abs(next_x - t[0]) + abs(next_y - t[1]) for t in treasures]
            closest_treasure = min(dists)
            dist_to_goal = closest_treasure
        else:
            dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])
            
        if my_hold:
             dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])

        # odleglosc od nagrody najblizszej
        max_dist = self.env.width + self.env.height
        features[3] = dist_to_goal / max_dist
        
        # czy podniesiony skarb
        if (next_x, next_y) in treasures and not my_hold:
            features[4] = 1.0
        
        # czy zdeponowanie skarbu
        if (next_x, next_y) == base_pos and my_hold:
            features[5] = 1.0
        

        possible_actions = self.env.get_possible_actions(state, self.agent_id)
        if possible_actions:
            temp_q_values = []
            for act in [LEFT, DOWN, RIGHT, UP]:
                if act in possible_actions:
                    temp_features = self._extract_basic_features(state, act)
                    temp_q = np.dot(self.weights[:6], temp_features[:6])
                    temp_q_values.append(temp_q)
                else:
                    temp_q_values.append(-1e10)
            
            temperature = 1.0
            exp_q = np.exp((np.array(temp_q_values) - np.max(temp_q_values)) / temperature)
            probs = exp_q / np.sum(exp_q)
            
            features[6] = probs[0]
            features[7] = probs[1]
            features[8] = probs[2]
            features[9] = probs[3]

        return features
    
    def _extract_basic_features(self, state, action):
        pos1, pos2, treasures, hold1, hold2 = state
        
        if self.agent_id == '1':
            my_pos = pos1
            op_pos = pos2
            my_hold = hold1
        else:
            my_pos = pos2
            op_pos = pos1
            my_hold = hold2
            
        dx, dy = ACTIONS[action]
        next_x = my_pos[0] + dx
        next_y = my_pos[1] + dy
        
        features = np.zeros(10)
        features[0] = 1.0
        
        if not (0 <= next_x < self.env.width and 0 <= next_y < self.env.height):
            features[1] = 1.0
            return features
            
        if self.env.map[next_y][next_x] == '#':
            features[1] = 1.0
            return features

        if self.env.map[next_y][next_x] == 'H':
            features[2] = 1.0
            
        base_pos = self.env.bases[self.agent_id]
        
        if my_hold:
            target_pos = base_pos
        elif treasures:
            dists = [abs(next_x - t[0]) + abs(next_y - t[1]) for t in treasures]
            closest_treasure = min(dists)
            dist_to_goal = closest_treasure
        else:
            dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])
            
        if my_hold:
             dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])

        max_dist = self.env.width + self.env.height
        features[3] = dist_to_goal / max_dist
        
        if (next_x, next_y) in treasures and not my_hold:
            features[4] = 1.0
            
        if (next_x, next_y) == base_pos and my_hold:
            features[5] = 1.0

        return features
    


    def get_q_value(self, state, action):
        features = self.extract_features(state, action)
        return np.dot(self.weights, features)

    def choose_action(self, state):
        possible_actions = self.env.get_possible_actions(state, self.agent_id)
        if not possible_actions:
            return 0 
            
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        q_values = [self.get_q_value(state, a) for a in possible_actions]
        max_q = max(q_values)
        
        best_indices = [i for i, q in enumerate(q_values) if q == max_q]
        best_idx = random.choice(best_indices)
        
        return possible_actions[best_idx]

    def learn(self, state, action, reward, next_state, done):
        features = self.extract_features(state, action)
        prediction = np.dot(self.weights, features)
        
        if done:
            target = reward
        else:
            next_actions = self.env.get_possible_actions(next_state, self.agent_id)
            if next_actions:
                next_q_values = [self.get_q_value(next_state, a) for a in next_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0
            target = reward + self.gamma * max_next_q
        
        error = target - prediction
        self.weights += self.alpha * error * features


class AgentWrapper:
    def __init__(self, agent):
            self.agent = agent
    def __contains__(self, state):
            return True
    def __getitem__(self, state):
            return self.agent.choose_action(state)