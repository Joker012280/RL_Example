import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import crr
import os
import sys

path = (os.path.abspath(os.path.join((os.path.dirname(__file__)),'..')))
sys.path.append(os.path.join(path,'TD3'))
import TD3




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

max_episode_num = 200
hidden = 128
## 결과값 프린트 주기
print_interval = 10

## Train
total_reward = 0
# online_agent = TD3.TD3(state_dim,hidden,action_dim)
# noise_generator = TD3.Noisegenerator(0,0.1)
offline_agent = crr.CRR(state_dim,hidden,action_dim)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
td3_path = "Td3.pth"
crr_path = "Crr_check.pth"
## 전에 사용했던 모델 가져오기
load = True
if load == True :
    # temp = torch.load(td3_path)
    # online_agent.load_state_dict(temp['model_state_dict'])
    # online_agent.eval()
    temp = torch.load(crr_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

# First Test for TD3
print("TD3 Testing")
# for num_episode in range(max_episode_num):
#     state = env.reset()
#     global_step = 0
#     done = False
#     reward = 0
#     while not done:
#         global_step += 1
#         state = torch.from_numpy(state).float()
#         action = online_agent.actor_network(state).item()
#         ## noise 추가
#         action += noise_generator.generate()
#
#         ## Action 값이 범위를 넘어서지 않도록 설정
#         action = max(min(action, 2.0), -2.0)
#
#         next_state, reward, done, _ = env.step([action])
#         ## Replay Buffer의 저장
#
#         state = next_state
#
#         total_reward += reward
#
#         if done:
#             break
#     ## 결과값 프린트
#     if num_episode % print_interval == 0 and num_episode != 0:
#         clear_output()
#
#         print("# of episode : {}, average score : {:.1f}".format(num_episode, \
#                                                                  total_reward / print_interval))
#         list_total_reward.append(total_reward / print_interval)
#         total_reward = 0.0
#
#


# Second Test for Crr
print("CRR Testing")
list_total_off_reward = []
for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.FloatTensor(state)
        action = offline_agent.actor_network(state)
        action = action.sample()

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action,min=-2,max=2)

        next_state, reward, done, _ = env.step(action.numpy())
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

plot_durations("Testing_Agent_tu1000_f.png")
