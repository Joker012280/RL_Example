import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cql
import os
import sys

path = (os.path.abspath(os.path.join((os.path.dirname(__file__)),'..')))
sys.path.append(os.path.join(path,'TD3'))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import TD3


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


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
env = gym.make('Pendulum-v0')
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
online_agent = TD3.TD3(state_dim,hidden,action_dim)
noise_generator = TD3.Noisegenerator(0,0.1)
offline_agent = cql.CQL(state_dim,hidden,action_dim)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
td3_path = "Td3.pth"
cql_path = "Cql.pth"
## 전에 사용했던 모델 가져오기
load = True
if load == True :
    temp = torch.load(td3_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    temp = torch.load(cql_path)
    offline_agent.load_state_dict(temp['model_state_dict'])
    offline_agent.eval()
    print("End Loading")

# First Test for TD3
print("TD3 Testing")
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

plot_durations("Testing_Agent_" +str(cql_path)+".png")


# Checking Q values

print(len(offline_agent.memory.memory))
online_q = []
offline_q = []
offline_adv = []
with torch.no_grad() :
    for i in range(len(offline_agent.memory.memory)) :
        st,at,_,_,_ = offline_agent.memory.memory[i]

        if i != 0 and i % 1 ==0 :
            st = st.unsqueeze(0)
            at = at.unsqueeze(0)
            q,q1 = online_agent.critic1_network(st,at).item(),online_agent.critic2_network(st,at).item()
            min_q = min(q, q1)
            print(min_q)
            online_q.append(float(min_q))
            # writer.add_scalar("OffT/Online Q1",min_q,i)
            q2,q3 = offline_agent.critic_network_1(st,at),offline_agent.critic_network_2(st,at)
            min_q2 = min(q2, q3)
            offline_q.append(float(min_q2))
            # writer.add_scalar("OffT/Offline Q1", min_q2, i)

            v = offline_agent.value_network(st)
            adv = min_q2 -v
            offline_adv.append(float(adv))
            # writer.add_scalar("OffT/Q-V",adv,i )
            # writer.add_scalar("OffT/Diff", diff, i)


    online_q_mean = np.mean(online_q)
    online_q_std = np.std(online_q)
    offline_q_mean = np.mean(offline_q)
    offline_q_std = np.std(offline_q)
    offline_adv_mean = np.mean(offline_adv)
    offline_adv_std = np.std(offline_adv)
    for i in range(len(online_q)) :
        online_q_normalized = (online_q[i] - online_q_mean) / online_q_std
        writer.add_scalar("OffT/Online Q1",online_q_normalized,i)
        offline_q_normalized = (offline_q[i] - offline_q_mean) / offline_q_std
        writer.add_scalar("OffT/Offline Q1", offline_q_normalized, i)
        offline_adv_normalized = (offline_adv[i] - offline_adv_mean) / offline_adv_std
        writer.add_scalar("OffT/Offline Adv",offline_adv_normalized,i )
        writer.add_scalar("OffT/Diff",online_q_normalized-offline_q_normalized,i)

    online_q = []
    offline_q = []
    offline_adv = []
    for i in range(len(online_agent.memory.memory)) :
        st,at,_,_,_ = online_agent.memory.memory[i]

        if i != 0 and i % 1 ==0 :
            st = st.unsqueeze(0)
            at = at.unsqueeze(0)
            q, q1 = online_agent.critic1_network(st, at).item(), online_agent.critic2_network(st, at).item()
            min_q = min(q, q1)
            online_q.append(float(min_q))
            # writer.add_scalar("OffT/Online Q1",min_q,i)
            q2, q3 = offline_agent.critic_network_1(st, at), offline_agent.critic_network_2(st, at)
            min_q2 = min(q2, q3)
            offline_q.append(float(min_q2))
            # writer.add_scalar("OffT/Offline Q1", min_q2, i)

            v = offline_agent.value_network(st)
            adv = min_q2 - v
            offline_adv.append(float(adv))
            # writer.add_scalar("OffT/Q-V",adv,i )
            # writer.add_scalar("OffT/Diff", diff, i)

    online_q_mean = np.mean(online_q)
    online_q_std = np.std(online_q)
    offline_q_mean = np.mean(offline_q)
    offline_q_std = np.std(offline_q)
    offline_adv_mean = np.mean(offline_adv)
    offline_adv_std = np.std(offline_adv)
    for i in range(len(online_q)):
        online_q_normalized = (online_q[i] - online_q_mean) / online_q_std
        writer.add_scalar("OnT/Online Q1", online_q_normalized, i)
        offline_q_normalized = (offline_q[i] - offline_q_mean) / offline_q_std
        writer.add_scalar("OnT/Offline Q1", offline_q_normalized, i)
        offline_adv_normalized = (offline_adv[i] - offline_adv_mean) / offline_adv_std
        writer.add_scalar("OnT/Offline Adv", offline_adv_normalized, i)
        writer.add_scalar("OnT/Diff", online_q_normalized - offline_q_normalized, i)