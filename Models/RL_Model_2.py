import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class PokerNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(PokerNN, self).__init__()
        # Define your network structure here
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = PokerNN(state_size, action_size)
        self.target_model = PokerNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)  # Experience replay memory
        self.epsilon = 1.0                  # Exploration factor
        self.epsilon_decay = 0.995          # Decay rate of epsilon
        self.epsilon_min = 0.01             # Minimum epsilon
        self.gamma = 0.99                   # Discount factor
        self.tau = 0.001                    # Target network soft update
        self.update_target_counter = 0      # Counter to update target network

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()
        else:
            return random.choice(range(self.action_size))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.learn(state, action, reward, next_state, done)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action]).unsqueeze(0)
        reward = torch.FloatTensor([reward]).unsqueeze(0)
        done = torch.FloatTensor([done]).unsqueeze(0)

        current_q = self.model(state).gather(1, action)
        max_next_q = self.target_model(next_state).detach().max(1)[0].unsqueeze(1)
        expected_q = reward + (self.gamma * max_next_q * (1 - done))

        loss = F.mse_loss(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_counter += 1
        if self.update_target_counter % 5 == 0:
            self.soft_update(self.model, self.target_model)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))