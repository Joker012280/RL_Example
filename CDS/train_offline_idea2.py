import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import Idea2
import os
import sac
import d4rl
from torch.utils.data import DataLoader, TensorDataset
import sys
import mujoco_py
import Buffer
import argparse
from distutils.util import strtobool

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


def plot_durations(name,total,average,behavior):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(total)
    durations_tt = torch.FloatTensor(average)
    plt.title('Training_'+str(args.algo))
    plt.xlabel('num of epoch / '+str(print_interval))
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(),label='Smooth')
    plt.plot(durations_tt.numpy(),label='Average')
    plt.axhline(y=behavior, color ='r', label="Behavior")
    plt.grid()
    plt.legend()
    plt.savefig(name)


parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--env", type=str, default="Walker2d-v3", help="Gym env name, default : Walker2d-v3")
# parser.add_argument("--task", type=str, default="run-forward",help="Specific Task, default : run-forward")
parser.add_argument("--print_interval", type=int, default = 2000)
parser.add_argument("--algo", type=str,default ="Idea2")
parser.add_argument("--epoch", type=int, default=500000, help="Number of epoch, default : 200000")
parser.add_argument("--gpu",type=str,default = "0")
parser.add_argument("--data",type=str,default="d4rl")

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")


## Environment
env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]

max_epoch = args.epoch

hidden = 256
print_interval = args.print_interval
weight_temperature = [5,float("inf")]
if args.data == "d4rl" :
    dataloader = prep_dataloader()
else :
    raise e 
## Train
agent = Idea2.Idea2(state_dim,
                  hidden,
                  action_dim,
                  action_bounds = action_bounds,
                  dataloader = dataloader,
                  device=device)    



print("Env : {} || Num Epoch : {} || Device : {} ".format(args.env,args.epoch,device))


def testing(agent,behavior_agent = False):
    ## Task state expansion
    max_episode_num = 5
    total_reward = 0
    for num_episode in range(max_episode_num):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            state = torch.FloatTensor(state).to(device)
            action = agent.actor_network.get_action(state)
            
            next_state, reward, done, info = env.step(action.cpu().detach().numpy())
                       
            state = next_state

            total_reward += reward

            if done:
                break
                
    env.close()
    
    return total_reward / max_episode_num

total_reward = 0
list_total_reward = []
list_average_reward = []
best_reward = float("-inf")
actor_loss = None

## Training
for epoch in range(1,max_epoch):               
    q_loss,actor_loss,behavior_loss,alpha= agent.train_net()
    ## Result Log
    if epoch % print_interval == 0 and epoch != 0:
        average_reward = testing(agent)
        if total_reward == 0 :
            total_reward = average_reward
        else :
            total_reward = 0.99 * total_reward + 0.01 * average_reward

        print("Epoch : {} || Episode Reward : {:.1f} || Running score : {:.1f} ".format(epoch,average_reward, total_reward))

        list_total_reward.append(total_reward)

        list_average_reward.append(average_reward)

        ## Save Best
        if best_reward < average_reward :
            best_reward = average_reward
            torch.save({
                'model_state_dict': agent.state_dict(),
            }, 'idea_result/'+str(args.algo)+'_best_Walker2d.pth')

        ## Plot Saving

        plot_durations("idea_result/"+str(args.algo)+".png",list_total_reward,list_average_reward,0)



        if actor_loss is not None:
            print("Critic Loss : {:.1f} || Actor Loss : {:.1f} || Behavior Loss : {:.1f} || Alpha Mean : {:.1f} ".format(q_loss,actor_loss,behavior_loss,alpha))

            
# Saving Model            
torch.save({
        'model_state_dict': agent.state_dict(),
    }, 'idea_result/'+str(args.algo)+'_Walker2d.pth')


    
## Testing Original
if args.data == "d4rl" :
    behavior_result = 2760.3
    plot_durations("idea_result/"+str(args.algo)+".png",list_total_reward,list_average_reward,behavior_result)
else : 
    raise e
print("End Training Offline Agent")

