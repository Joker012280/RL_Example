import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cql
import os
import sys


from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


def plot_durations(name):
    plt.figure(2)
    #plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    durations_t_off = torch.FloatTensor(list_total_off_reward)
    plt.title('Testing')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(),label='Online')
    plt.plot(durations_t_off.numpy(),label='Offline')
    plt.grid()
    plt.legend()

    plt.savefig(name)


## Environment
env = gym.make('Pendulum-v1')
## Action이 연속적이라 env.action_space.n을 사용하지않음.
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

max_episode_num = 250
hidden = 256
## 결과값 프린트 주기
print_interval = 10

## Train
total_reward = 0
online_agent = cql.CQL(state_dim,hidden,action_dim)

offline_agent = cql.CQL(state_dim,hidden,action_dim)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
cql_on_path = "CQL_ONLINE.pth"
cql_off_path = "Cql.pth"
## 전에 사용했던 모델 가져오기
load = True
if load == True :
    temp = torch.load(cql_on_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    temp = torch.load(cql_off_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

# First Test for TD3
print("CQL Testing")

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action,_ = online_agent.actor_network.evaluate(state)
        ## noise 추가


        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action,min=-2,max=2)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        ## Replay Buffer의 저장

        state = next_state

        total_reward += reward

        if done:
            break
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        clear_output()

        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)
        total_reward = 0.0




# Second Test for Crr
print("CQL Testing")
total_reward = 0
list_total_off_reward = []
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
        action = torch.clamp(action,min=-2,max=2)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        ## Replay Buffer의 저장

        state = next_state

        total_reward += reward

        if done:
            break
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_off_reward.append(total_reward / print_interval)
        total_reward = 0.0

plot_durations("Testing_Agent.png")

