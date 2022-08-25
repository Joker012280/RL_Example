import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import iql
import os
import sys

path = (os.path.abspath(os.path.join((os.path.dirname(__file__)), '..')))
sys.path.append(os.path.join(path, 'TD3'))
import TD3
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def plot_durations(name, list1, list2, list3, list4):
    plt.figure(2)
    # plt.clf()
    durations_t = torch.FloatTensor(list1)
    durations_t_off = torch.FloatTensor(list2)
    durations_t_off_5 = torch.FloatTensor(list3)
    durations_t_off_9 = torch.FloatTensor(list4)
    plt.title('Testing')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(), label='Online')
    plt.plot(durations_t_off.numpy(), label='Offline_7')
    plt.plot(durations_t_off_5.numpy(), label='Offline_5')
    plt.plot(durations_t_off_9.numpy(), label='Offline_9')
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
online_agent = TD3.TD3(state_dim, hidden, action_dim)
noise_generator = TD3.Noisegenerator(0, 0.1)
offline_agent = iql.IQL(state_dim, hidden, action_dim)
offline_agent_5 = iql.IQL(state_dim, hidden, action_dim)
offline_agent_9 = iql.IQL(state_dim, hidden, action_dim)

## 전에 사용했던 모델 있는 곳
td3_path = "Td3.pth"
iql_path = "IQL_mid-expert_0.7.pth"
iql_path_5 = "IQL_mid-expert_0.5.pth"
iql_path_9 = "IQL_mid-expert_0.9.pth"
## 전에 사용했던 모델 가져오기
load = True
if load == True:
    temp = torch.load(td3_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    temp = torch.load(iql_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    temp = torch.load(iql_path_5)
    offline_agent_5.load_state_dict(temp['model_state_dict'])
    offline_agent_5.eval()
    temp = torch.load(iql_path_9)
    offline_agent_9.load_state_dict(temp['model_state_dict'])
    offline_agent_9.eval()

    print("End Loading")

# First Test for TD3
print("TD3 Testing")
list_total_reward = []
for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action = online_agent.actor_network(state).item()
        ## noise 추가
        action += noise_generator.generate()

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = max(min(action, 2.0), -2.0)

        next_state, reward, done, _ = env.step([action])
        ## Replay Buffer의 저장
        online_agent.memory.push((state, torch.FloatTensor([action]), torch.FloatTensor([reward]), \
                                  torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

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
print("IQL Testing")
list_total_off_reward = []

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.FloatTensor(state)
        _, _, action = offline_agent.actor_network(state)

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action, min=-2, max=2)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        ## Replay Buffer의 저장
        offline_agent.memory.push((state, torch.FloatTensor([action]), torch.FloatTensor([reward]), \
                                   torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
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

print("IQL Testing")

list_total_off_reward_5 = []

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.FloatTensor(state)
        _, _, action = offline_agent_5.actor_network(state)

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action, min=-2, max=2)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        state = next_state

        total_reward += reward

        if done:
            break
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_off_reward_5.append(total_reward / print_interval)
        total_reward = 0.0

print("IQL Testing")
list_total_off_reward_9 = []

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.FloatTensor(state)
        _, _, action = offline_agent_9.actor_network(state)

        ## Action 값이 범위를 넘어서지 않도록 설정
        action = torch.clamp(action, min=-2, max=2)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        state = next_state

        total_reward += reward

        if done:
            break
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_off_reward_9.append(total_reward / print_interval)
        total_reward = 0.0

plot_durations("Testing_Agent.png", list_total_reward, list_total_off_reward, list_total_off_reward_5,
               list_total_off_reward_9)

# Checking Q values


print(len(offline_agent.memory.memory))
online_q = []
offline_q = []
offline_q_5 = []
offline_q_9 = []
offline_adv = []
with torch.no_grad():
    for i in range(len(offline_agent.memory.memory)):
        st, at, _, _, _ = offline_agent.memory.memory[i]

        if i != 0 and i % 1 == 0:
            st = st.unsqueeze(0)
            at = at.unsqueeze(0)
            q, q1 = online_agent.critic1_network(st, at).item(), online_agent.critic2_network(st, at).item()
            min_q = min(q, q1)
            online_q.append(float(min_q))
            q2, q3 = offline_agent.critic_network_1(st, at), offline_agent.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q.append(float(min_q2))

            v = offline_agent.value_network(st)
            adv = min_q2 - v
            offline_adv.append(float(adv))

            q2, q3 = offline_agent_5.critic_network_1(st, at), offline_agent_5.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q_5.append(float(min_q2))

            q2, q3 = offline_agent_9.critic_network_1(st, at), offline_agent_9.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q_9.append(float(min_q2))


    online_q_mean = np.mean(online_q)
    online_q_std = np.std(online_q)
    offline_q_mean = np.mean(offline_q)
    offline_q_std = np.std(offline_q)
    offline_q_5_mean = np.mean(offline_q_5)
    offline_q_5_std = np.std(offline_q_5)
    offline_q_9_mean = np.mean(offline_q_9)
    offline_q_9_std = np.std(offline_q_9)
    offline_adv_mean = np.mean(offline_adv)
    offline_adv_std = np.std(offline_adv)
    online_q_normalized = []
    offline_q_normalized = []
    offline_q_5_normalized = []
    offline_q_9_normalized = []
    for i in range(len(online_q)):
        online_q_normalized.append((online_q[i] - online_q_mean) / online_q_std)
        offline_q_normalized.append((offline_q[i] - offline_q_mean) / offline_q_std)
        offline_q_5_normalized.append((offline_q_5[i] - offline_q_5_mean) / offline_q_5_std)
        offline_q_9_normalized.append((offline_q_9[i] - offline_q_9_mean) / offline_q_9_std)
    for i in range(len(online_q)):
        writer.add_scalars("OffT/Q-val", {'On-Q': online_q_normalized[i], 'Q-7': offline_q_normalized[i],
                                          'Q-9': offline_q_9_normalized[i], 'Q-5': offline_q_5_normalized[i]}, i)
    print("Offline Finished")
    online_q = []
    offline_q = []
    offline_q_5 = []
    offline_q_9 = []
    offline_adv = []
    for i in range(len(online_agent.memory.memory)):
        st, at, _, _, _ = online_agent.memory.memory[i]

        if i != 0 and i % 1 == 0:
            st = st.unsqueeze(0)
            at = at.unsqueeze(0)
            q, q1 = online_agent.critic1_network(st, at).item(), online_agent.critic2_network(st, at).item()
            min_q = min(q, q1)
            online_q.append(float(min_q))
            q2, q3 = offline_agent.critic_network_1(st, at), offline_agent.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q.append(float(min_q2))

            v = offline_agent.value_network(st)
            adv = min_q2 - v
            offline_adv.append(float(adv))
            q2, q3 = offline_agent_5.critic_network_1(st, at), offline_agent_5.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q_5.append(float(min_q2))

            q2, q3 = offline_agent_9.critic_network_1(st, at), offline_agent_9.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q_9.append(float(min_q2))

    online_q_mean = np.mean(online_q)
    online_q_std = np.std(online_q)
    offline_q_mean = np.mean(offline_q)
    offline_q_std = np.std(offline_q)
    offline_q_5_mean = np.mean(offline_q_5)
    offline_q_5_std = np.std(offline_q_5)
    offline_q_9_mean = np.mean(offline_q_9)
    offline_q_9_std = np.std(offline_q_9)
    offline_adv_mean = np.mean(offline_adv)
    offline_adv_std = np.std(offline_adv)
    online_q_normalized = []
    offline_q_normalized = []
    offline_q_5_normalized = []
    offline_q_9_normalized = []

    for i in range(len(online_q)):
        online_q_normalized.append((online_q[i] - online_q_mean) / online_q_std)
        offline_q_normalized.append((offline_q[i] - offline_q_mean) / offline_q_std)
        offline_q_5_normalized.append((offline_q_5[i] - offline_q_5_mean) / offline_q_5_std)
        offline_q_9_normalized.append((offline_q_9[i] - offline_q_9_mean) / offline_q_9_std)
    for i in range(len(online_q)):
        writer.add_scalars("OnT/Q-val", {'On-Q': online_q_normalized[i], 'Q-7': offline_q_normalized[i],
                                         'Q-9': offline_q_9_normalized[i], 'Q-5': offline_q_5_normalized[i]}, i)
    print("Online Finished")