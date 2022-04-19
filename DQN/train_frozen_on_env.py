import gym
import gym_gridworlds
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os
import sys
import dqn

path = (os.path.abspath(os.path.join((os.path.dirname(__file__)),'..')))
sys.path.append(os.path.join(path,'TD3'))
import TD3

def plot_durations(name):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    plt.title('Training')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy())
    plt.savefig(name)

desc=["SFFF", "FFFF", "FFFF", "FFFG"]
## Environment
env = gym.make('FrozenLake-v1',desc = desc,map_name="4x4",is_slippery=False)
state_dim = env.observation_space.n
action_dim = env.action_space.n

max_episode_num = 25000
hidden = 64
## 결과값 프린트 주기
print_interval = 100
batch_size = 128
## Train
total_reward = 0
online_agent = dqn.DQN(state_dim,hidden,action_dim,batch_size=batch_size)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
dql_path = 'dqn_On_frozen.pth'
## 전에 사용했던 모델 가져오기
load = False
if load == True :
    temp = torch.load(dql_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    print("End Loading")

def state_modi(state) :
    state_arr = np.zeros(state_dim)
    state_arr[state] = 1
    state_arr = np.reshape(state_arr,[1,state_dim])
    return state_arr

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    state = state_modi(state)
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action = online_agent.select_action(state)
        next_state, reward, done, _ = env.step(action.detach().item())
        reward = reward -1
        next_state = state_modi(next_state)
        ## Replay Buffer의 저장
        online_agent.memory.push((state, action, torch.FloatTensor([reward]),\
                           torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

        state = next_state

        total_reward += reward

        if done :
            break
        ## Memory size가 커지고 나서 학습시작
        if online_agent.memory.size() > 500:
            online_agent.train_net()
    if num_episode !=0 and num_episode % 10 == 0 :
        online_agent.hard_load_dict()
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        clear_output()

        print("# of episode : {}, average score : {}".format(num_episode, \
                                                                 total_reward/print_interval))
        list_total_reward.append(total_reward / print_interval)
        total_reward = 0.0


## 모델 저장하기 !
torch.save({
    'model_state_dict': online_agent.state_dict(),
}, 'dqn_On_frozen.pth')

plot_durations("Online_Agent.png")
online_agent.memory.save_data("Frozen_online")
print("Finish Data Saving")



clear_output()
print("End Training Online Agent")

print("End Training!")