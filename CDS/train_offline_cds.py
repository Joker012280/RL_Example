import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cds
import os
import sys

# from torch.utils.tensorboard import SummaryWriter



###################### TODO ########################


## Environment
env = gym.make('Walker2d-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
hidden = 256

## Train
total_reward = 0
offline_agent = cds.CDS(state_dim,hidden,action_dim,task="run-forward")
list_total_reward = []

offline_agent.memory.load_data("run-forward_None")
offline_agent.memory_2.load_data("jump_None")

print("Data size : ",offline_agent.memory.size())
print("Data size : ",offline_agent.memory_2.size())
print("Finished Data Loading")


print("Start Training Offline Agent")
max_offline_train_num = 30000
print_interval = 500
model_path = None
target_update_interval = 1000
load = False
if load == True :
    temp = torch.load(cql_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

def testing():
    max_episode_num = 20
    total_reward = 0
    for num_episode in range(max_episode_num):
        state = env.reset()
        global_step = 0
        done = False
        reward = 0
        while not done:
            global_step += 1
            state = torch.FloatTensor(state)
            action,_ = offline_agent.actor_network.evaluate(state)

            ## Action 값이 범위를 넘어서지 않도록 설정
            action = torch.clamp(action, min=-2, max=2)

            next_state, reward, done, _ = env.step(action.detach().numpy())
            ## Replay Buffer의 저장

            state = next_state

            total_reward += reward

            if done:
                break
    return total_reward / max_episode_num


for train_num in range(max_offline_train_num):
    critic_loss_1,critic_loss_2,actor_loss = offline_agent.train_net()
    if train_num % target_update_interval == 0 and train_num != 0:
        offline_agent.load_dict()
    ## 결과값 프린트
    if train_num % print_interval == 0 and train_num != 0:
        print("# of train num : {}".format(train_num))
        average_reward = testing()
        print("Testing While Training : {} / Average Reward : {}".format(train_num,average_reward))

## 모델 저장하기 !
torch.save({
    'model_state_dict': offline_agent.state_dict(),
}, 'Cql.pth')

print("End Training!")