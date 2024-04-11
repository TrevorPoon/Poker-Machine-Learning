import torch
import torch.nn as nn
import torch.optim as optim

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
        self.model = PokerNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # ... Initialize other components of your learning algorithm
    
    def select_action(self, state):
        # Convert state to torch tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action probabilities from model
        action_probs = self.model(state_tensor)
        
        # Select an action based on probabilities
        action = torch.argmax(action_probs).item()
        
        return action
    
    def learn(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])

        # Compute Q values for current state
        pred = self.model(state)

        # Compute the Q value corresponding to the chosen action
        q_val = pred.gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute the expected Q values from the next state
        next_pred = self.model(next_state)
        next_q_val = next_pred.max(1)[0].detach()

        # Compute the target Q value
        expected_q_val = reward + (0.99 * next_q_val * (1 - int(done)))  # assuming a discount factor gamma of 0.99

        # Compute the loss as the mean squared error between the target Q value and the predicted Q value
        loss = F.mse_loss(q_val, expected_q_val)

        # Backpropagation and optimization
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update weights

        # Return the loss value to track performance if needed
        return loss.item()