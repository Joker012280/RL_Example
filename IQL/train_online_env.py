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
# path = (os.path.abspath(os.path.join((os.path.dirname(__file__)),'..')))
# sys.path.append(os.path.join(path,'TD3'))
# import TD3

def plot_durations(name):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    plt.title('Training')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy())
    plt.savefig(name)


## Environment
env = gym.make('Pendulum-v1')
## Action이 연속적이라 env.action_space.n을 사용하지않음.
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


max_episode_num = 500
hidden = 256
## 결과값 프린트 주기
print_interval = 10

## Train
total_reward = 0
online_agent = iql.IQL(state_dim,hidden,action_dim)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
iql_path = None
## 전에 사용했던 모델 가져오기
load = False
if load == True :
    temp = torch.load(iql_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    print("End Loading")


for num_episode in range(max_episode_num):
    state = env.reset()

    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action,_ = online_agent.actor_network.evaluate(state)

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action, min=-2, max=2)
        next_state, reward, done, _ = env.step(action.detach().numpy())

        ## Replay Buffer의 저장
        online_agent.memory.push((state, action.unsqueeze(1).detach(), torch.FloatTensor([reward]),\
                           torch.FloatTensor(next_state), torch.FloatTensor([done])))

        state = next_state

        total_reward += reward

        if done:
            break
        ## Memory size가 커지고 나서 학습시작
        if online_agent.memory.size() > 1000:
            online_agent.train_net()

    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        clear_output()

        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)
        total_reward = 0.0

plot_durations("Online_Agent.png")
online_agent.memory.save_data("online")
print("Finish Data Saving")

## 모델 저장하기 !
torch.save({
    'model_state_dict': online_agent.state_dict(),
}, 'IQL_Online.pth')

clear_output()
print("End Training Online Agent")

print("End Training!")