import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import torch.nn.functional as F
import os
import sys
from torch.distributions import Categorical
import mujoco_py
import Buffer
import torch.nn as nn
import d4rl
# import mujoco_py
import os
import random
import gym
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import argparse


# k = 4
# t = [0 for i in range(2)]

# # print(t)
# parser = argparse.ArgumentParser(description='RL')
# parser.add_argument('--n', nargs='+', default=[])
# args = parser.parse_args()

# a = torch.FloatTensor([1,2,-1,3,4,-5])
# b = torch.FloatTensor([1,1,1,1,1,1])
# c = 5 * a + 3 * b
# print(c)

# mu = 5
# sigma = 1

# s = np.random.noraml(mu,sigma,1000)
# expectile = 0.9
# quantile = 0.9
    

# print(args.n)
# a = torch.FloatTensor([[0,0,0],[1,1,1]])
# b = torch.FloatTensor([[4,4,4],[2,2,2]])
# c = torch.FloatTensor([[-1,-1,-1],[0,0,0]])
# print(a.mean(0))
# print(a.var(0))

# tt = torch.stack([a,b,c])
# print(tt)
# print(tt.size())
# print(tt.mean(0).size())
# print(tt.var(0))
# t = {0 : [1,2] , 1 : [3,4], 2 : [3,33]}
# print(t.keys())
# print(len(t.keys()))

# print(args.n)
# for i,name in enumerate(args.n) :
#     print(type(name))
#     print(i)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# t = torch.tensor([])
# tt = torch.tensor([1,2,3])
# print(tt)
# tt.to(device)
# print(tt)
# tt = tt.to(device)

# def plot_durations(name="check.png",total=None):
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.FloatTensor(total)
#     # durations_tt = torch.FloatTensor(average)
#     plt.title('Training_')
#     plt.xlabel('num of epoch / ')
#     plt.ylabel('reward')
#     plt.plot(durations_t.numpy(),label='Smooth')
#     # plt.plot(durations_tt.numpy(),label='Average')
#     plt.grid()
#     plt.legend()
#     plt.savefig(name)
    
# tt =[[0],[0],[0]]
# for i in range(10):
#     if tt[(i%3)][0] == 0:
#         tt[(i%3)][0] = 3
#     tt[(i%3)].append(i+3)
# print(tt)
# for i in range(3) :
#     plot_durations(name = "11"+str(i),total=tt[i])
# for i in range(3):
#     print(random.randint(0,2))

# def cat_sample(*samples) :

#     state, action, jump_reward,reward,backward_reward, next_state, done = zip(*samples)
#     state = torch.cat(state)
#     state = state.reshape(5, -1)
#     next_state = torch.cat(next_state)
#     next_state = next_state.reshape(5, -1)
#     action = torch.cat(action)
#     action = action.reshape(5,-1)
#     reward = torch.cat(reward)
#     reward = reward.unsqueeze(1)
#     done = torch.cat(done)
#     done = done.unsqueeze(1)
#     return state, action, jump_reward,reward,backward_reward, next_state, done

# buff = Buffer.Replay_buffer(3)
# print(buff.size())
# print(buff.memory)
# buff.push((1,2,3))
# buff.push((1,2,3))
# buff.push((1,2,3))
# buff.push((1,2,3))
# print("-----------------")
# print(buff.size())
# print(buff.memory)
# # print(buff.sample())
# sample = buff.sample()
# a,b,c = zip(*sample)
# print(a)
# print(b)
# print(c)



# buff.load_data("jump_expert")

# samples_1=buff.sample()
# state_1, action_1, jump_reward_1,forward_reward_1,backward_reward_1, next_state_1, done_1 = cat_sample(*samples_1)
# print(state_1)
# print(state_1.size())
# tt,t = torch.chunk(state_1,2,dim=0)
# print(tt)
# print(tt.size())
# print(t)
# print(t.size())
# xxx = torch.cat([tt,t],dim=0)
# print(xxx)
# print(xxx.size())
print(d4rl.ope.get_returns("walker2d-medium"))
# env = gym.make("walker2d-medium-v2")
# print(env.get_returns("walker2d-medium"))
# dataset = env.get_returns()
# print(dataset)
# def prep_dataloader(env_id="walker2d-medium-v2", batch_size=1, seed=1):
#     env = gym.make(env_id)
#     dataset = env.get_dataset()
#     print(dataset.keys())
#     tensors = {}
#     for k, v in dataset.items():
#         if k in ["actions", "observations", "next_observations", "rewards", "terminals","timeouts"]:
#             if  k is not "terminals":
#                 tensors[k] = torch.from_numpy(v).float()
#             else:
#                 tensors[k] = torch.from_numpy(v).long()

#     tensordata = TensorDataset(tensors["observations"],
#                                tensors["actions"],
#                                tensors["rewards"][:, None],
#                                tensors["next_observations"],
#                                tensors["terminals"][:, None],
#                                tensors["timeouts"][:,None])
#     dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
#     return dataloader
# def check() :
#     experience = next(iter(dataloader))
#     print(experience)
# dataloader = prep_dataloader()
# for i in range(20) :
#     check()

# print("-------------------------------")
# experience = next(iter(dataloader))
# print(experience)
# print("-------------------------------")
# experience = next(iter(dataloader))
# print(experience)
# print("-------------------------------")
# check()
# print("-------------------------------")
# check()
# print("-------------------------------")

# for batch_idx, experience in enumerate(dataloader):
#         states, actions, rewards, next_states, dones = experience
#         states = states
#         actions = actions
#         rewards = rewards
#         next_states = next_states
#         dones = dones
#         print("-------------------------------")
#         print(states,actions,rewards,next_states)
#         print("-------------------------------")
#         print(batch_idx)
#         print("-------------------------------")
# def init_weight(layer, initializer="he normal"):
#     if initializer == "xavier uniform":
#         nn.init.xavier_uniform_(layer.weight)
#     elif initializer == "he normal":
#         nn.init.kaiming_normal_(layer.weight)
# class PolicyNet(nn.Module):
#     def __init__(self):
#         super(PolicyNet, self).__init__()

#         self.fc1 = nn.Linear(16, 48)
#         # init_weight(self.fc1)
#         self.fc1.bias.data.zero_()
#         self.fc2 = nn.Linear(48, 48)
#         # init_weight(self.fc2)
#         self.fc2.bias.data.zero_()
#         self.fc3 = nn.Linear(48, 4)  # Prob of left
#         # init_weight(self.fc3,"xavier uniform")
#         self.fc3.bias.data.zero_()
        
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

# def state_modi(state) :
#     state_arr = np.zeros(state_dim)
#     state_arr[state] = 1
#     state_arr = np.reshape(state_arr,[1,state_dim])
#     return state_arr

# desc=["SSSS", "SSSS", "SSSS", "SSSS"]
# ## Environment
# env = gym.make('FrozenLake-v1',desc = desc,map_name="4x4",is_slippery=False)
# state_dim = env.observation_space.n
# action_dim = env.action_space.n

# Q = np.zeros([state_dim,action_dim])


# learning_rate = 0.4
# discount_factor = 0.95
# num_episodes = 5000
# # policy_net = PolicyNet()
# # optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

# list_total_reward = []
# # state_pool = []
# # action_pool = []
# # reward_pool = []
# print("-------------------------------------------------")
# # t = torch.eye(state_dim)
# # tt = torch.ones(state_dim)
# # for i in range(state_dim) :
# #     state = t[i]
# #     probs = policy_net(state)
# #     print(probs.detach())
# #     # print(m)
# # probs = policy_net(tt)
# # print(probs.detach())
# for i in range(num_episodes) :
#     # state_pool = []
#     # action_pool = []
#     # reward_pool = []
#     # loss_pool = []
#     state = env.reset()
#     total_reward = 0
#     done = False
#     step = 0
#     while step < 100:
#         step += 1
#         action = np.argmax(Q[state, :] + np.random.randn(1, action_dim) * (1. / (i + 1)))
#         # state = state_modi(state)
#         # state = torch.FloatTensor(state)
#         # probs = policy_net(state)
#         # m = Categorical(probs)
#         # action = m.sample()
#         # action = action.data.numpy().astype(int)[0]
#         # print(action)
#         next_state,reward,done,_ = env.step(action)
#         # reward = reward -1
#         reward = -1
#         Q[state,action] = Q[state,action] + learning_rate*(reward + discount_factor \
#                                                            * np.max(Q[next_state,:]) - Q[state,action])
# #         state_pool.append(state)
# #         action_pool.append(float(action))
# #         reward_pool.append(reward)
# #         total_reward += reward
# #         state = next_state

# #     optimizer.zero_grad()
    
    
# #     ## 현재 에피소드의 Step을 통해 loss 계산
# #     for i in range(step):
# #         state = state_pool[i]
# #         action = torch .FloatTensor([action_pool[i]])
# #         reward = reward_pool[i]
        


# #         probs = policy_net(state)
# #         m = Categorical(probs)
# #         loss = -m.log_prob(action) * reward  ## Negtive score function x reward
# #         loss_pool.append(loss)
# #         loss.backward()

# #     ## Optimize
# #     optimizer.step()
#     # print(sum(loss_pool)/len(loss_pool))
#     ## Trajectory 다시 초기화

#     list_total_reward.append(total_reward)
# print("-------------------------------------------------")
# # t = torch.eye(state_dim)
# # for i in range(state_dim) :
# #     state = t[i]
# #     probs = policy_net(state)
# #     print(probs.detach())
# #     # print(m)
# # probs = policy_net(tt)
# # print(probs.detach())
# print("Q - Table")
# for i in range(action_dim) :
#     print("For action : ",i)
#     print(np.round(Q[:,i],4).reshape(4,4))
#     x = np.arange(-0.5,3,1)
#     y = np.round(Q[:,0],4).reshape(4,4)
#     plt.pcolormesh(x,x,y)
#     plt.savefig("Testing"+str(i)+".png")

# x = np.arange(-0.5,3,1)
# y = np.round(Q[:,0],4).reshape(4,4)
# plt.pcolormesh(x,x,y)
# plt.savefig("Testing.png")