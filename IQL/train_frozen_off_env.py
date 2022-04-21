import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import iql
import os
import sys

from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

desc=["SFFF", "FFFF", "FFFF", "FFFG"]
## Environment
env = gym.make('FrozenLake-v1',desc = desc,map_name="4x4",is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n
hidden = 64

## Train
total_reward = 0
expectile = 0.9
temperature = 10.0
offline_agent = iql.IQL(state_dim,hidden,action_dim,expectile=expectile,temperature= temperature,is_discrete=True)
list_total_reward = []

offline_agent.memory.load_data("Frozen_online")
print("Finished Data Loading")
print("Data size : ",offline_agent.memory.size())

print("Start Training Offline Agent")
max_offline_train_num = 250000
print_interval = 10000
iql_path = "IQL_off_"+str(temperature)+"_"+str(expectile)+".pth"
iql_path = None

if iql_path is not None :
    temp = torch.load(iql_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

def state_modi(state) :
    state_arr = np.zeros(state_dim)
    state_arr[state] = 1
    state_arr = np.reshape(state_arr,[1,state_dim])
    return state_arr

def testing():
    with torch.no_grad():
        max_episode_num = 20
        total_reward = 0
        for num_episode in range(max_episode_num):
            state = env.reset()
            global_step = 0
            done = False
            reward = 0
            state = state_modi(state)
            while not done:
                global_step += 1
                state = torch.FloatTensor(state)
                action,_ = offline_agent.actor_network.evaluate(state)
                next_state, reward, done, _ = env.step(action.item())
                reward = reward -1
                next_state = state_modi(next_state)
                ## Replay Buffer의 저장

                state = next_state

                total_reward += reward

                if done:
                    break
    return total_reward / max_episode_num


for train_num in range(max_offline_train_num):
    critic_loss_1,critic_loss_2,actor_loss,_ = offline_agent.train_net()
    if train_num % print_interval == 0 and train_num != 0:
        clear_output()
        print("# of train num : {}".format(train_num))
        average_reward = testing()
        print("Testing While Training : {} / Average Reward : {}".format(train_num,average_reward))

## 모델 저장하기 !
torch.save({
    'model_state_dict': offline_agent.state_dict(),
}, 'IQL_off_'+str(temperature)+"_"+str(expectile)+'.pth')

print("End Training!")



