import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import crr
import os
import sys

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
path = (os.path.abspath(os.path.join((os.path.dirname(__file__)),'..')))
sys.path.append(os.path.join(path,'pendulum_data'))

## Environment
env = gym.make('Pendulum-v1')
## Action이 연속적이라 env.action_space.n을 사용하지않음.
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
hidden = 128

## Train
total_reward = 0
offline_agent = crr.CRR(state_dim,hidden,action_dim)
list_total_reward = []

offline_agent.memory.load_data("C:/Users/user/Desktop/RL_Example/pendulum_data/mid-expert")
print("Finished Data Loading")
print("Data size : ",offline_agent.memory.size())

print("Start Training Offline Agent")
max_offline_train_num = 300000
print_interval = 1000
crr_path = "Crr.pth"
target_update_interval = 1000
load = False
if load == True :
    temp = torch.load(crr_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

for train_num in range(max_offline_train_num):
    critic_loss,actor_loss = offline_agent.train_net()
    writer.add_scalar("Critic Loss",critic_loss,train_num)
    writer.add_scalar("Actor Loss",actor_loss,train_num)
    if train_num % target_update_interval == 0 and train_num != 0:
        offline_agent.load_dict()
    ## 결과값 프린트
    if train_num % print_interval == 0 and train_num != 0:
        clear_output()
        print("# of train num : {}".format(train_num))

## 모델 저장하기 !
torch.save({
    'model_state_dict': offline_agent.state_dict(),
}, 'Crr_check.pth')

print("End Training!")