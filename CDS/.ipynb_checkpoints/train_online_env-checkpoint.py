import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
# import cds
import cql
import sac
import sac_q
import os
import sys
import mujoco_py
import Buffer
import argparse

def plot_durations(name):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    durations_tt = torch.FloatTensor(list_average_reward)
    plt.title('Training_'+str(args.task))
    plt.xlabel('Num of epoch / '+str(print_interval))
    plt.ylabel('reward')
    plt.plot(durations_t.numpy(),label='Smooth')
    plt.plot(durations_tt.numpy(),label='Average')
    plt.grid()
    plt.legend()
    plt.savefig(name)

parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--env", type=str, default="Walker2d-v3", help="Gym env name, default : Walker2d-v3")
parser.add_argument("--task", type=str, default="run-forward",help="Specific Task, default : run-forward")
parser.add_argument("--algo", type=str, default="sac")
parser.add_argument("--print_interval", type=int, default = 1000)
parser.add_argument("--episode", type=int, default=20000, help="Number of epoch, default : 20000")
parser.add_argument("--data_type", type=str, default="medium-replay")
parser.add_argument("--reward_scale", type=float, default=5)
parser.add_argument("--gpu",type=str,default = "0")

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
print("Env : {} || Task : {} || Num Episode : {} || Device : {} || {} ".format(args.env,args.task,args.episode,device,torch.cuda.device_count()))
## Task checkning


## Environment

if args.task == "jump" :
    healthy_z_range = (0.8,10)
    env = gym.make(args.env,healthy_z_range = healthy_z_range)

else :
    env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]

max_episode_num = args.episode
batch_size = 256
hidden = 256
print_interval = args.print_interval
saving_buffer = Buffer.Replay_buffer(batch_size,capacity = None)


## Train
total_reward = 0
if args.algo == "sac":
    online_agent = sac.SAC(state_dim,hidden,action_dim,action_bounds=action_bounds,reward_scale = args.reward_scale,device=device)
elif args.algo == "cql":
    online_agent = cql.CQL(state_dim,hidden,action_dim,reward_scale = args.reward_scale,device=device)
elif args.algo == "sac_q":
    online_agent = sac_q.SAC(state_dim,hidden,action_dim,action_bounds=action_bounds,reward_scale = args.reward_scale,device=device)
    
list_total_reward = []
list_average_reward = []
best_reward = float("-inf")

# if args.data_type == "expert" :
#     max_episode_num = 5
#     print_interval = 1
#     model_path = "results/SAC_jump_Walker2d_best.pth"
#     temp = torch.load(model_path)
#     online_agent.load_state_dict(temp['model_state_dict'])
#     online_agent.eval()
#     print("End Loading")


actor_loss = None
average_reward = 0
for num_episode in range(1,max_episode_num+1):
    state = env.reset()
    init_z= env.sim.data.qpos[1]
    done = False
    reward = 0
    episode_reward = 0
    while not done:
        state = torch.from_numpy(state).float()
        action = online_agent.actor_network.get_action(state.to(device))
        next_state, reward, done, info = env.step(action.detach().cpu().numpy())
        
        
        jump_reward = -(abs(info["x_velocity"]))-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
        forward_reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
        backward_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
        
        
        if args.task == "jump" :
            reward = jump_reward
        elif args.task == "run-forward" :
            reward = forward_reward
        elif args.task == "run-backward" :
            reward = backward_reward
            
        ## Buffer
        online_agent.memory.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([reward]),\
                           torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

        # Train
        if online_agent.memory.size() >= batch_size:# and args.data_type != "medium":
            q1_loss,q2_loss,actor_loss,value_loss = online_agent.train_net()
         
        state = next_state
        episode_reward += reward
        
    if num_episode == 0 :
        total_reward = episode_reward
    else :
        total_reward = 0.99 * total_reward + 0.01 * episode_reward

    average_reward += episode_reward
    ## Result Log
    if num_episode % print_interval == 0 and num_episode != 0:
        
        print("Num of Episode : {} || Episode Reward : {:.1f} || Running score : {:.1f} ".format(num_episode, episode_reward, total_reward))
        list_total_reward.append(total_reward)
        list_average_reward.append(average_reward / print_interval)
        if average_reward > best_reward : 
            best_reward = average_reward
            torch.save({
                'model_state_dict': online_agent.state_dict(),
            }, 'results/Online'+str(args.algo)+'_'+str(args.task)+'best_Walker2d.pth')
        average_reward = 0
        if actor_loss is not None:
            print("Policy Loss : {:.1f} || Q1 Loss : {:.1f} || Q2 Loss : {:.1f} || Value Loss : {:.1f} || Buffer Length : {} ".format(actor_loss,q1_loss,q2_loss,value_loss,online_agent.memory.size()))

## Model Save

torch.save({
        'model_state_dict': online_agent.state_dict(),
    }, 'results/Online'+str(args.algo)+'_'+str(args.task)+'_Walker2d.pth')

plot_durations("results/Online_Agent_"+str(args.task)+'_'+str(args.algo)+".png")
#online_agent.memory.save_data("online")
# saving_buffer.save_data(str(args.task)+"_"+str(args.data_type))



print("End Training Online Agent")

