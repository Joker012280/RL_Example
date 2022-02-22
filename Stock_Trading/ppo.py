import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self,state_dim,hidden,action_dim):
        super(Model,self).__init__()

        self.actor_layer = nn.Sequential(
            nn.Linear(state_dim,hidden),
            nn.Tanh(),
            nn.Linear(hidden,128),
            nn.Tanh(),
            nn.Linear(128,action_dim),
            nn.Softmax(),

        )
        self.critic_layer = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden,128),
            nn.Tanh(),
            nn.Linear(128, 1),

        )



    def forward(self):
        raise NotImplementedError

    def act(self,state,mask=None):

        probs = self.actor_layer(state)
        if mask is not None :
            mask = torch.FloatTensor([mask])
            probs = probs * mask

        print("Action Probs : ",probs)
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



class PPO(nn.Module):

    def __init__(self,state_dim,action_dim,hidden,memory,device,actor_learning_rate,eps,testing=False):
        super(PPO, self).__init__()


        self.network = Model(state_dim,hidden,action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr = actor_learning_rate)
        self.old_network = Model(state_dim,hidden,action_dim)
        self.old_network.load_state_dict(self.network.state_dict())
        self.memory = memory
        self.eps = eps
        self.mse = nn.MSELoss()
        self.epoch = 5

    def train_net(self):

        if self.memory.size() == 0 :
            return

        old_actions,old_prob,state, next_state, reward, done = self.memory.sample()


        for _ in range(self.epoch) :
            log_prob,entropy,state_value = self.network.evaluate(state,old_actions)

            ratio = torch.exp(log_prob - old_prob.detach())

            advantages = reward - state_value.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantages
            loss = - torch.min(surr1,surr2) + 0.5*self.mse(state_value,reward) - 0.2*entropy
            print("Total Loss : ",loss.mean().item())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_network.load_state_dict(self.network.state_dict())


class Replay_buffer():
    def __init__(self, max_size=1000):
        self.memory = []
        self.max_size = max_size
        self.position = 0
        self.buffer_size = 0

    def push(self, data):
        if len(self.memory) == self.max_size:
            self.memory[int(self.position)] = data
            self.position = (self.position + 1) % self.max_size
        else:
            self.memory.append(data)
            self.buffer_size += 1

    def sample(self):
        old_actions,probs,states= torch.FloatTensor(),torch.FloatTensor(), torch.FloatTensor()
        next_states, rewards, done = [], [], []

        ## 받은 샘플들을 합쳐서 내보냄
        for i in range(self.buffer_size):
            old_action,prob,state,next_state, reward, done_ = self.memory[i]
            old_actions = torch.cat((old_actions,old_action))
            probs = torch.cat((probs,prob))
            states = torch.cat([states,state])
            next_states.append([next_state])
            rewards.append([reward])
            done.append([done_])

        next_states = torch.FloatTensor(next_states)
        ## Return 값이 각 요소에 따른 텐서들을 전달
        states = states.view(-1,23)
        return old_actions.detach(),probs,states.detach(),  \
               next_states, torch.FloatTensor(rewards), torch.FloatTensor(done)

    def clear(self):
        self.memory = []
        self.position = 0
        self.buffer_size = 0

    def size(self):
        return self.buffer_size
