from player import DQN, QTrainer
from snake import SnakeEnv
import torch


def trainer(num_episodes, max_steps_per_episode, env, _device, _lr = 0.001, _gamma= 0.99):

    model = DQN(env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n)

    model.to(_device)
    trainer = QTrainer(model,lr=_lr,gamma=_gamma,device=_device)

    for episode in range(num_episodes):
        print(f'EPOCH:{episode}')
        state = env.reset()
        for step in range(max_steps_per_episode):
            action = trainer.get_action(state.reshape(-1),epsilon=0.1)
            next_state, reward, done, _ = env.step(action)
            loss = trainer.train(state, action, reward, next_state, done)

            if done:
                break

            state = next_state

        if (episode+1) % 10000 == 0:
            model.save_model(f'model_{episode+1}.pth')