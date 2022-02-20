import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable
import plotting
import Buffer


## Hyperparameter
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
batch_size = 50
discount_factor = 0.99
eps = 0.2
max_episode_num = 300

## 결과값 프린트 주기
print_interval = 1

env = gym.make('MountainCar-v0').unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


class Model(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Model,self).__init__()

        self.actor_layer = nn.Sequential(
            nn.Linear(state_dim,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self,state):
        probs = self.actor_layer(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_prob = dist.log_prob(action)
        return action,action_prob

    def evaluate(self,state,action):
        probs = self.actor_layer(state)
        dist = Categorical(probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        value = self.critic_layer(state)

        return action_logprob, dist_entropy, value




def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    m = torch.min(ratio * advantage, clipped)
    return -m



class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()


        self.network = Model(state_dim,action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr = actor_learning_rate)
        self.old_network = Model(state_dim,action_dim)
        self.old_network.load_state_dict(self.network.state_dict())

        self.mse = nn.MSELoss()
        self.epoch = 5

    def train_net(self):

        if memory.size() == 0 :
            return

        old_actions,old_prob,state, next_state, reward, done = memory.sample()


        for _ in range(self.epoch) :
            log_prob,entropy,state_value = self.network.evaluate(state,old_actions)

            ratio = torch.exp(log_prob - old_prob.detach())

            advantages = reward - state_value.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,1-eps,1+eps) * advantages
            loss = - torch.min(surr1,surr2) + 0.5*self.mse(state_value,reward) - 0.2*entropy
            # print("Total Loss : ",loss.mean().item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_network.load_state_dict(self.network.state_dict())


## Train
total_reward = 0
max_reward = float('-inf')

agent = PPO()

memory = Buffer.Replay_buffer()
list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'ppo_mountaincar_best.pth'
## 전에 사용했던 모델 가져오기
load = False
if load == True:
    temp = torch.load(PATH)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()


for num_episode in range(max_episode_num):
    state = env.reset()
    step = 0
    done = False
    reward = 0
    prev_action_prob = None
    while not done:
        step += 1
        state = torch.FloatTensor([state])
        action,action_prob = agent.old_network.act(state)
        mem_action = action
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action[0])

        ## Trajectory
        if(step >1 ) :
            memory.push((mem_action, action_prob,state ,next_state, reward, done))

        state = next_state

        total_reward += reward

        prev_action_prob = action_prob


        ## Memory size가 커지고 나서 학습시작
        if (memory.size() == batch_size) and (memory.size() != 0):
            agent.train_net()
            memory.clear()

        if done:
            agent.train_net()
            memory.clear()
            break


    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)
        if max_reward < total_reward :
            torch.save({
                'model_state_dict': agent.state_dict(),
            }, 'ppo_mountaincar_best.pth')
            max_reward = total_reward
        total_reward = 0.0

plt.plot(list_total_reward)
plt.show()
