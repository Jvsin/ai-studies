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
        
        # --- DEFINICJA CECH (FEATURES) ---
        features = np.zeros(10)
        
        # Cecha 0: Bias (zawsze 1, pozwala przesunąć wykres funkcji)
        features[0] = 1.0
        
        # Sprawdzamy, czy ruch jest w ogóle legalny (czy nie wyjdzie poza mapę/w ścianę)
        if not (0 <= next_x < self.env.width and 0 <= next_y < self.env.height):
            # Ruch poza mapę - bardzo źle
            features[1] = 1.0 # "Is Invalid Move"
            return features # Reszta cech nie ma znaczenia
            
        if self.env.map[next_y][next_x] == '#':
            # Ruch w ścianę - bardzo źle
            features[1] = 1.0 # "Is Invalid Move"
            return features

        # Cecha 2: Czy wchodzę w dziurę (H)?
        if self.env.map[next_y][next_x] == 'H':
            features[2] = 1.0
            
        # Cecha 3: Dystans do celu (znormalizowany)
        # Ustal cel: Jeśli mam skarb -> Baza. Jeśli nie -> Najbliższy skarb.
        target_pos = None
        base_pos = self.env.bases[self.agent_id]
        
        if my_hold:
            target_pos = base_pos
        elif treasures:
            # Znajdź najbliższy skarb od PRZYSZŁEJ pozycji
            # (Żeby agent widział, że ruch w jego stronę zmniejsza dystans)
            dists = [abs(next_x - t[0]) + abs(next_y - t[1]) for t in treasures]
            closest_treasure = min(dists)
            # Wirtualny cel to pozycja tego skarbu (dla wizualizacji), tu liczymy dystans
            dist_to_goal = closest_treasure
        else:
            # Brak skarbów, wracaj do bazy
            dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])
            
        if my_hold:
             dist_to_goal = abs(next_x - base_pos[0]) + abs(next_y - base_pos[1])

        # Normalizacja: Dzielimy przez obwód mapy, żeby wartości były małe (np. 0.0 - 1.0)
        max_dist = self.env.width + self.env.height
        features[3] = dist_to_goal / max_dist
        
        # Cecha 4: Czy ten ruch PODNOSI skarb?
        # (Jesteśmy w next_pos, sprawdzamy czy tam jest skarb i czy nie trzymamy)
        if (next_x, next_y) in treasures and not my_hold:
            features[4] = 1.0
            
        # Cecha 5: Czy ten ruch ODNOSI skarb do bazy?
        if (next_x, next_y) == base_pos and my_hold:
            features[5] = 1.0
        
        # Cechy 6-9: Prawdopodobieństwo wyboru każdej akcji (softmax z Q-wartości)
        # Obliczamy Q dla wszystkich możliwych akcji z OBECNEJ pozycji
        possible_actions = self.env.get_possible_actions(state, self.agent_id)
        if possible_actions:
            # Obliczamy Q-wartości dla wszystkich możliwych akcji
            temp_q_values = []
            for act in [LEFT, DOWN, RIGHT, UP]:
                if act in possible_actions:
                    # Rekurencyjnie obliczamy Q dla tej akcji (bez prawdopodobieństw, żeby uniknąć cykliczności)
                    temp_features = self._extract_basic_features(state, act)
                    temp_q = np.dot(self.weights[:6], temp_features[:6])  # Używamy tylko pierwszych 6 wag
                    temp_q_values.append(temp_q)
                else:
                    temp_q_values.append(-1e10)  # Bardzo niska wartość dla niedostępnych akcji
            
            # Softmax z temperaturą (im mniejsza temperatura, tym bardziej deterministyczne)
            temperature = 1.0
            exp_q = np.exp((np.array(temp_q_values) - np.max(temp_q_values)) / temperature)
            probs = exp_q / np.sum(exp_q)
            
            # Przypisujemy prawdopodobieństwa do cech 6-9
            features[6] = probs[0]  # LEFT
            features[7] = probs[1]  # DOWN
            features[8] = probs[2]  # RIGHT
            features[9] = probs[3]  # UP

        return features
    
    def _extract_basic_features(self, state, action):
        """Pomocnicza metoda do wyodrębnienia podstawowych cech bez prawdopodobieństw akcji"""
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
        """
        Oblicza Q(s, a) jako iloczyn skalarny wag i cech.
        Q(s, a) = w0*f0 + w1*f1 + ... + wn*fn
        """
        features = self.extract_features(state, action)
        return np.dot(self.weights, features)

    def choose_action(self, state):
        """Strategia Epsilon-Greedy"""
        possible_actions = self.env.get_possible_actions(state, self.agent_id)
        if not possible_actions:
            return 0 
            
        # Eksploracja
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Eksploatacja (Wybierz akcję z największym Q)
        q_values = [self.get_q_value(state, a) for a in possible_actions]
        max_q = max(q_values)
        
        # Jeśli jest kilka akcji o tym samym max_q, wylosuj jedną z nich (żeby nie faworyzować kierunków)
        best_indices = [i for i, q in enumerate(q_values) if q == max_q]
        best_idx = random.choice(best_indices)
        
        return possible_actions[best_idx]

    def learn(self, state, action, reward, next_state, done):
        """
        Aktualizacja wag (Gradient Descent).
        w <- w + alpha * [Target - Prediction] * features
        """
        # 1. Obliczamy Prediction: Q(s, a)
        features = self.extract_features(state, action)
        prediction = np.dot(self.weights, features)
        
        # 2. Obliczamy Target: R + gamma * max Q(s', a')
        if done:
            target = reward
        else:
            # Szukamy max Q dla następnego stanu
            next_actions = self.env.get_possible_actions(next_state, self.agent_id)
            if next_actions:
                next_q_values = [self.get_q_value(next_state, a) for a in next_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0
            target = reward + self.gamma * max_next_q
        
        # 3. Obliczamy błąd (TD Error)
        error = target - prediction
        
        # 4. Aktualizacja wag
        # Gradient funkcji Q względem wag to po prostu wektor cech (features)
        self.weights += self.alpha * error * features