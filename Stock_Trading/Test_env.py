import pandas as pd
import torch
import Customenv
import matplotlib.pyplot as plt
import numpy as np
import ppo

## Modify if you use cuda (GPU)
# device = torch.device("cuda" if torch.cuda.isavailable() else "cpu")
device = "cpu"

## Hyperparameter
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
batch_size = 15
discount_factor = 0.99
eps = 0.2
epoch = 30

## Env Dataset
dataset = pd.read_csv("Dataset2.csv")
env = Customenv.stock_env(dataset,0,50000)

state_dim = env.observation_space
action_dim = env.action_space
hidden = 256

total_reward = 0
max_reward = float('-inf')


## Agent
memory = ppo.Replay_buffer()
agent = ppo.PPO(state_dim,action_dim,hidden,memory,device,actor_learning_rate,eps,testing = True)

list_total_reward = []


## 전에 사용했던 모델 있는 곳
PATH = 'stock_model_best.pth'
## 전에 사용했던 모델 가져오기
load = True
if load == True:
    temp = torch.load(PATH)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()
    print("Finish Model Loading")

for num_episode in range(epoch) :
    state,reward,done,info = env.reset()
    reward = 0
    done = False
    step = 0
    while not done :
        step += 1
        state = torch.FloatTensor(np.array(state))
        action, action_prob = agent.old_network.act(state,info[0])
        action = action.detach().cpu().numpy()
        next_state, reward, done, info = env.step(action)

        state = next_state
        total_reward += reward

    ## 결과값 프린트
    print("# of episode : {}, average score : {:.1f}".format(num_episode,total_reward))
    list_total_reward.append(total_reward)
    total_reward = 0.0

plt.plot(list_total_reward)
plt.show()
