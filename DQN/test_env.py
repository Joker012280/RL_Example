import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import dqn
import os
import sys


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def plot_durations(name, list1):
    plt.figure(2)
    # plt.clf()
    durations_t = torch.FloatTensor(list1)

    plt.title('Testing')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(), label='Online')
    plt.grid()
    plt.legend()

    plt.savefig(name)

desc=["SFFF", "FFFF", "FFFF", "FFFG"]
## Environment
env = gym.make('FrozenLake-v1',desc = desc,map_name="4x4",is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n

max_episode_num = 3
hidden = 64
## 결과값 프린트 주기
print_interval = 1
batch_size = 32
## Train
total_reward = 0
online_agent = dqn.DQN(state_dim,hidden,action_dim,batch_size=batch_size)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
dqn_path = 'dqn_On_frozen.pth'
## 전에 사용했던 모델 가져오기
load = True
if load == True :
    temp = torch.load(dqn_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    print("End Loading")

def state_modi(state) :
    state_arr = np.zeros(state_dim)
    state_arr[state] = 1
    state_arr = np.reshape(state_arr,[1,state_dim])
    return state_arr

for i in range(16) :
    state = i
    state = state_modi(state)
    print("Q - val : ",online_agent.actor_network(torch.from_numpy(state).float()))
    print("Action : ",online_agent.select_det_action(torch.from_numpy(state).float()))

for num_episode in range(max_episode_num):
    state = env.reset()
    state = state_modi(state)
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action= online_agent.select_det_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = reward -1
        next_state = state_modi(next_state)

        state = next_state

        total_reward += reward

        if done :
            break

    ## 결과값 프린트
    if num_episode % print_interval == 0 :
        clear_output()

        print("# of episode : {}, average score : {}".format(num_episode, \
                                                                 total_reward/print_interval))
        list_total_reward.append(total_reward / print_interval)
        total_reward = 0.0

plot_durations("Online_Agent_test.png",list_total_reward)
