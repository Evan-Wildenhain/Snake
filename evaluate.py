from player import DQN, QTrainer
from snake import SnakeEnv
from training import trainer
import torch
import time

def evaluate(model, env, num_episodes = 100, max_steps_per_episode = 100000):
    for episode in range(num_episodes):
        state = env.reset()
        for j in range(max_steps_per_episode):
            state_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=-1).item()
            state,reward,done, _ = env.step(action)
            env.render()
            time.sleep(0.5)
            if done:
                break

env = SnakeEnv()
num_episodes = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer(num_episodes,500,env, device)


model = DQN(env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n)
model.load_state_dict(torch.load(f'model_10000.pth'))
model.to(device)

evaluate(model,env,num_episodes=10)
