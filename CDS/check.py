import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import os
import sys
import mujoco_py


import mujoco_py
import os
import random

# xx = np.array([[3.,4.,5.]])
# x = np.array([[1.,0.,0.]])
# print(x)
# tt = np.repeat(xx,5,axis=0)
# t = np.repeat(x,5,axis=0)
# print(t)
# print(t.shape)
# tt = torch.from_numpy(tt)
# t = torch.from_numpy(t)
# j = torch.cat([t,tt],axis=1)
# print(j.shape)
# print(j)

# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# print(sim.data.qpos)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# sim.step()
# print(sim.data.qpos)
# # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
# #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
# #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
# #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
# #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
# #  -2.22862221e-05]


# ## Environment
# env = gym.make('Walker2d-v3')
# ## Action이 연속적이라 env.action_space.n을 사용하지않음.
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])

# max_episode_num = 400
# hidden = 256
# ## 결과값 프린트 주기
# print_interval = 10

# ## Train
# total_reward = 0
# online_agent = cql.CQL(state_dim,hidden,action_dim)
# list_total_reward = []

# ## Start Training For TD3 Agent(Stacking Buffer)
# for num_episode in range(max_episode_num):
#     state = env.reset()
#     global_step = 0
#     done = False
#     reward = 0
#     while not done:
#         global_step += 1
#         state = torch.from_numpy(state).float()
#         action,_ = online_agent.actor_network.evaluate(state)

#         ## Action 값이 범위를 넘어서지 않도록 설정
#         action = torch.clamp(action, min=-2, max=2)

#         next_state, reward, done, _ = env.step(action.detach().numpy())
#         ## Replay Buffer의 저장
#         online_agent.memory.push((state, action.detach().unsqueeze(1), torch.FloatTensor([reward]),\
#                            torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

#         state = next_state

#         total_reward += reward

#         if done:
#             break
#         ## Memory size가 커지고 나서 학습시작
#         if online_agent.memory.size() > 1000:
#             online_agent.train_net()
#     ## 결과값 프린트
#     if num_episode % print_interval == 0 and num_episode != 0:
#         clear_output()

#         print("# of episode : {}, average score : {:.1f}".format(num_episode, \
#                                                                  total_reward / print_interval))
#         list_total_reward.append(total_reward / print_interval)
#         total_reward = 0.0

# plot_durations("Online_Agent.png")
# online_agent.memory.save_data("online")
# print("Finish Data Saving")

# ## 모델 저장하기 !
# torch.save({
#     'model_state_dict': online_agent.state_dict(),
# }, 'CQL_ONLINE.pth')

# clear_output()

# print("End Training Online Agent")

