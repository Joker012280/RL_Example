import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import iql
import os
import mujoco_py
import sys
import bc

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()


## Environment
env = gym.make('Walker2d-v3')
## Action이 연속적이라 env.action_space.n을 사용하지않음.
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
hidden = 256

## Train
total_reward = 0
expectile = 0.7
temperature = 3.0
offline_agent = iql.IQL(state_dim,hidden,action_dim,expectile=expectile,temperature= temperature)
# offline_agent_2 = bc.BC(state_dim,hidden,action_dim)
list_total_reward = []


offline_agent.memory.load_data('run-forward_expert')
# offline_agent_2.memory.load_data('run-forward_expert')
print("Finished Data Loading")
print("Data size : ",offline_agent.memory.size())
# print("Data size : ",offline_agent_2.memory.size())

print("Start Training Offline Agent")
max_offline_train_num = 1000000
print_interval = 1000
iql_path = "IQL_Online.pth"
load = False

if load == True :
    temp = torch.load(iql_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")


def testing():
    max_episode_num = 5
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
            # action,_ = offline_agent_2.actor_network.evaluate(state)

            next_state, reward, done, info = env.step(action.detach().numpy())
            
            reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            state = next_state

            total_reward += reward

            if done:
                break
    env.close()
    return total_reward / max_episode_num


for train_num in range(max_offline_train_num):
    critic_loss_1,critic_loss_2,actor_loss,_ = offline_agent.train_net()
    
    # actor_loss = offline_agent_2.train_net()
    # writer.add_scalar("Critic Loss 1",critic_loss_1,train_num)
    # writer.add_scalar("Critic Loss 2", critic_loss_2, train_num)
    # writer.add_scalar("Actor Loss",actor_loss,train_num)
    ## 결과값 프린트
    if train_num % print_interval == 0 and train_num != 0:
        clear_output()
        print("# of train num : {}".format(train_num))
        average_reward = testing()
        print("Testing While Training : {} / Average Reward : {}".format(train_num,average_reward))

## 모델 저장하기 !
torch.save({
    'model_state_dict': offline_agent.state_dict(),
}, 'IQL_Walker'+str(expectile)+'.pth')

# torch.save({
#     'model_state_dict': offline_agent_2.state_dict(),
# }, 'BC_Walker'+str(expectile)+'.pth')

print("End Training!")



