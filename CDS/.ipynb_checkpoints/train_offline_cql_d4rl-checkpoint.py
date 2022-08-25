import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
# import cds
import cql
import os
import sys
import mujoco_py
import Buffer
import argparse
import d4rl
from torch.utils.data import DataLoader, TensorDataset

def plot_durations(name):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    durations_tt = torch.FloatTensor(list_average_reward)
    plt.title('Training_'+str(args.task))
    plt.xlabel('num of epoch / '+str(print_interval))
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(),label='Smooth')
    plt.plot(durations_tt.numpy(),label='Average')
    plt.grid()
    plt.legend()
    plt.savefig(name)



parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--env", type=str, default="Walker2d-v3", help="Gym env name, default : Walker2d-v3")
parser.add_argument("--task", type=str, default="run-forward",help="Specific Task, default : run-forward")
parser.add_argument("--print_interval", type=int, default = 1000)
parser.add_argument("--algo", type=str,default ="cql")
parser.add_argument("--epoch", type=int, default=200000, help="Number of epoch, default : 200000")
parser.add_argument("--reward_scale", type=float, default=5)
parser.add_argument("--gpu",type=str,default = "0")

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
print("Env : {} || Task : {} || Num Episode : {} || Device : {} || {} ".format(args.env,args.task,args.episodes,device,torch.cuda.device_count()))

def prep_dataloader(env_id="walker2d-medium-v2", batch_size=256, seed=1):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if  k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    return dataloader
    

## Environment
env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]
dataloader = prep_dataloader()
max_epoch = args.epoch
batch_size = 256
hidden = 256
print_interval = args.print_interval


## Train
# self, state_dim, hidden, action_dim, tau=None, target_entropy=None, temperature=None,batch_size = 256,reward_scale=1,device=None)
agent = cql.CQL(state_dim,hidden,action_dim,action_bounds=action_bounds,dataloader = dataloader,reward_scale = args.reward_scale,device=device)    
# agent.memory.load_data("run-forward_medium-replay")


print("Memory Size : {}".format(agent.memory.size()))

actor_loss = None
def testing():
    max_episode_num = 20
    total_reward = 0
    for num_episode in range(max_episode_num):
        state = env.reset()
        init_z= env.sim.data.qpos[1]
        global_step = 0
        done = False
        reward = 0
        while not done:
            global_step += 1
            state = torch.FloatTensor(state)
            action = agent.actor_network.get_action(state.to(device))
            
            next_state, reward, done, info = env.step(action.cpu().detach().numpy())
            
            ## Task Reward
            jump_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
            forward_reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            backward_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            
            if args.task == "jump" :
                reward = jump_reward
            elif args.task == "run-forward" :
                reward = forward_reward
            elif args.task == "run-backward" :
                reward = backward_reward
        
            
            state = next_state

            total_reward += reward

            if done:
                break
    return total_reward / max_episode_num


total_reward = 0
list_total_reward = []
list_average_reward = []
for epoch in range(1,max_epoch):               
    q1_loss,q2_loss,actor_loss = agent.train_net()
    ## Result Log
    if epoch % print_interval == 0 and epoch != 0:
        average_reward = testing()
        if total_reward == 0 :
            total_reward = average_reward
        else :
            total_reward = 0.99 * total_reward + 0.01 * average_reward
        print("Epoch : {} || Episode Reward : {:.1f} || Running score : {:.1f} ".format(epoch, average_reward, total_reward))
        list_total_reward.append(total_reward)
        list_average_reward.append(average_reward)
        if actor_loss is not None:
            print("Policy Loss : {:.1f} || Q1 Loss : {:.1f} || Q2 Loss : {:.1f}".format(actor_loss,q1_loss,q2_loss))
        

# Saving Model            
torch.save({
        'model_state_dict': agent.state_dict(),
    }, 'cql_results/'+str(args.algo)+'_'+str(args.task)+'_d4rl_final_Walker2d.pth')

plot_durations("cql_results/Offline_Agent_"+str(args.algo)+'_'+str(args.task)+"_d4rl.png")
print("End Training Offline Agent")

