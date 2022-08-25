import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cql
import sac
import os
import sys
import mujoco_py
import argparse


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--env", type=str, default="Walker2d-v3", help="Gym env name, default : Walker2d-v3")
parser.add_argument("--task", type=str, default="run-forward",help="Specific Task, default : run-forward")
parser.add_argument("--algo", type=str,default ="cql")
parser.add_argument("--model_path", type=str,default = None)
parser.add_argument("--epoch", type=int, default=200000, help="Number of epoch, default : 200000")
parser.add_argument("--reward_scale", type=float, default=5)
parser.add_argument("--gpu",type=str,default = "0")

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
print("Env : {} || Task : {} || Num Episode : {} || Device : {} || {} ".format(args.env,args.task,args.episodes,device,torch.cuda.device_count()))
def plot_durations(name):
    plt.figure(2)
    #plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    plt.title('Testing')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy())
    plt.grid()
    # plt.legend()

    plt.savefig(name)


## Environment
env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]

max_episode_num = 250
hidden = 256


## Train
total_reward = 0
agent = cql.CQL(state_dim,hidden,action_dim,action_bounds=action_bounds)
list_total_reward = []

## 전에 사용했던 모델 있는 곳
agent_path = args.model_path
## 전에 사용했던 모델 가져오기

if agent_path is not None :
    temp = torch.load(agent_path)
    agent.load_state_dict(temp['model_state_dict'])
    agent.eval()
    print("End Loading")

for num_episode in range(max_episode_num):
    state = env.reset()
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action= agent.actor_network.get_action(state)
    
        next_state, reward, done, _ = env.step(action.detach().numpy())
        ## Replay Buffer의 저장

        state = next_state

        total_reward += reward

        if done:
            break
    print("# of episode : {}, average score : {:.1f}".format(num_episode, \
                                                                 total_reward))
    list_total_reward.append(total_reward)
    total_reward = 0.0


plot_durations("cql_results/Testing_Agent_"+str(args.algo)+'_'+str(args.task)+".png")
print("End Training Offline Agent")