import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_size)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class MyAgent:
    def __init__(self, input_dims=18, n_actions=5, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_dec=0.995, epsilon_min=0.05):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.action_space = [0, 1, 2, 3, 4] # Forward, Backward, Left, Right, Stop

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = "records/race_model.pth"

        # Policy Network i Target Network
        self.policy_net = DuelingDQN(input_dims, n_actions).to(self.device)
        self.target_net = DuelingDQN(input_dims, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(50000)
        self.best_reward = -float('inf')

    def normalize_state(self, state):
        walls = np.array(state[0], dtype=np.float32) / 1000.0 
        cars = np.array(state[1], dtype=np.float32) / 200.0
        velocity = np.array([state[3] / 10.0], dtype=np.float32) 

        return np.concatenate([walls, cars, velocity])

    def choose_action(self, state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.policy_net(state_tensor)
            return torch.argmax(actions).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(self.normalize_state(state), action, reward, self.normalize_state(next_state), done)

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

        # Obliczanie Q-values dla bieżących stanów
        q_values = self.policy_net(states).gather(1, actions)

        # Obliczanie Target Q-values (Double DQN logic albo zwykły DQN)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping dla stabilności
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(self.policy_net.state_dict(), self.path)
            print(f"Model saved with reward: {reward}")

    def load(self):
        if os.path.exists(self.path):
            self.policy_net.load_state_dict(torch.load(self.path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Model loaded.")
            self.epsilon = 0.05 # Jeśli ładujemy do gry, minimalna eksploracja