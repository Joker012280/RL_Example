import gym
import torch
import torch.nn as nn
import Buffer
import torch.optim as optim
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.isavailable() else "cpu")

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    m = torch.min(ratio * advantage, clipped)
    return -m



## Hyperparameter
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
batch_size = 50
discount_factor = 0.99
eps = 0.2
test_num = 100

## 결과값 프린트 주기
print_interval = 1

env = gym.make('MountainCar-v0').unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


total_reward = 0
max_reward = float('-inf')

agent = PPO()

memory = Buffer.Replay_buffer()
list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'ppo_mountaincar_best.pth'
## 전에 사용했던 모델 가져오기
load = True
if load == True:
    temp = torch.load(PATH)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()


for num_episode in range(test_num) :
    state = env.reset()
    reward = 0
    done = False
    step = 0
    while not done :
        step += 1
        state = torch.FloatTensor(np.array(state))
        action, action_prob = agent.old_network.act(state)
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)

        state = next_state

        total_reward += reward

        ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)

        total_reward = 0.0

plt.plot(list_total_reward)
plt.show()
