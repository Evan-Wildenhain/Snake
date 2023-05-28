import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save_model(self, file_name = "model.pth"):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self,model,lr,gamma, device):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.device = device

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)

        self.eval = nn.MSELoss()

    def train(self,state,action,reward,next_state,game_over):
        state = torch.tensor(state,dtype=torch.float).view(-1).to(self.device)
        next_state = torch.tensor(next_state,dtype=torch.float).view(-1).to(self.device)
        action = torch.tensor(action,dtype=torch.long).to(self.device)
        reward = torch.tensor(reward,dtype=torch.float).to(self.device)
        game_over = torch.tensor(game_over,dtype=torch.bool).to(self.device)

        predicted_q_values = self.model(state)
        predicted_q_value = predicted_q_values[action]

        next_q = self.model(next_state).max()

        target_q = reward
        if not game_over:
            target_q = reward + self.gamma *next_q

        loss = self.eval(predicted_q_value,target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([i for i in range(self.model.output_dim)])
        else:
            state = torch.tensor(state,dtype=torch.float).view(-1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()
        
