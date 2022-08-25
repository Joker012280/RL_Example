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
    plt.xlabel('num of episodes / '+str(print_interval))
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
parser.add_argument("--print_interval", type=int, default = 100)
parser.add_argument("--epoch", type=int, default = 500)
parser.add_argument("--episodes", type=int, default = 20000, help="Number of episodes, default : 20000")
parser.add_argument("--data_type", type=str, default="medium-replay")
parser.add_argument("--reward_scale", type=float, default=5)
parser.add_argument("--gpu",type=str,default = "0")

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
print("Env : {} || Task : {} || Num Episode : {} || Device : {} || {} ".format(args.env,args.task,args.episodes,device,torch.cuda.device_count()))

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

max_episode_num = args.episodes
batch_size = 256
hidden = 256
if args.task == "jump" :
    epoch = -0.4
else :
    epoch = 0
max_epoch = args.epoch
print_interval = args.print_interval
saving_buffer = Buffer.Replay_buffer(batch_size,capacity = None)


## Train
total_reward = 0
online_agent = sac.SAC(state_dim,hidden,action_dim,action_bounds=action_bounds,reward_scale = args.reward_scale,device=device)
    
list_total_reward = []
list_average_reward = []
best_reward = float("-inf")


actor_loss = None
episode_count = 0
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
        
        
        ## Dataset Saving for different Tasks
        # if args.data_type == "medium-replay" and epoch < 100: 
        if args.data_type == "medium-replay" and epoch < 100 and epoch > 0: 
            saving_buffer.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([jump_reward]),torch.FloatTensor([forward_reward]),torch.FloatTensor([backward_reward]),\
                               torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
        if args.data_type == "medium" and epoch > 100 :
            saving_buffer.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([jump_reward]),torch.FloatTensor([forward_reward]),torch.FloatTensor([backward_reward]),\
                               torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
        if args.data_type == "expert" and epoch > 500 :
            saving_buffer.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([jump_reward]),torch.FloatTensor([forward_reward]),torch.FloatTensor([backward_reward]),\
                               torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
        
        ## Corresponding Reward Function
        if args.task == "jump" :
            reward = jump_reward
        elif args.task == "run-forward" :
            reward = forward_reward
        elif args.task == "run-backward" :
            reward = backward_reward
            
        ## Buffer for Tranining
        online_agent.memory.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([reward]),\
                           torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

        
        ## Train
        if online_agent.memory.size() >= batch_size and epoch < 500 :
            epoch += 0.001
            if args.data_type == "medium" and epoch > 100 :
                pass
            else :
                q1_loss,q2_loss,actor_loss,value_loss = online_agent.train_net()
         
        state = next_state
        episode_reward += reward
        
    ## Terminate For loop When condition is satisfied for each Task
    if epoch >= max_epoch :
        episode_count += 1
        if args.data_type == "expert" and episode_count > 5 :
            break
    if args.data_type == "medium" and epoch > 100:
        episode_count += 1
        if episode_count == 500 :
            break
    if args.data_type == "medium-replay" and epoch > 100 :
        break
    
    ## Episode Running Score
    if num_episode == 0 :
        total_reward = episode_reward
    else :
        total_reward = 0.99 * total_reward + 0.01 * episode_reward
    average_reward += episode_reward

    ## Result Log
    if num_episode % print_interval == 0 and num_episode != 0:
        
        print("Num of episode : {} || Episode Reward : {:.1f} || Running score : {:.1f} || Epoch : {:.4f} ".format(num_episode, episode_reward, total_reward,epoch))
        list_total_reward.append(total_reward)
        list_average_reward.append(average_reward / print_interval)
        average_reward = 0
        if actor_loss is not None:
            print("Policy Loss : {:.1f} || Q1 Loss : {:.1f} || Q2 Loss : {:.1f} || Value Loss : {:.1f} || Buffer Length : {} ".format(actor_loss,q1_loss,q2_loss,value_loss,online_agent.memory.size()))
        if num_episode % 500 == 0:
            torch.save({
                            'model_state_dict': online_agent.state_dict(),
                        }, 'results/'+str(args.algo)+'_'+str(args.task)+'_'+str(args.data_type)+'_Walker2d.pth')


## Plotting
plot_durations("results/Online_Agent_"+str(args.task)+'_'+str(args.data_type)+".png")

## Saving Buffer
print("Saving Data : ",saving_buffer.size())
saving_buffer.save_data(str(args.task)+"_"+str(args.data_type))

print("Finish Data Saving")

## Model Save

torch.save({
        'model_state_dict': online_agent.state_dict(),
    }, 'results/'+str(args.algo)+'_'+str(args.task)+'_'+str(args.data_type)+'_last_Walker2d.pth')



# if args.data_type =="expert" :
#     agent_path = "./results/sac_"+str(args.task)+"_expert_Walker2d.pth"
#     if agent_path is not None :
#         temp = torch.load(agent_path)
#         online_agent.load_state_dict(temp['model_state_dict'])
#         online_agent.eval()
#         print("End Loading")
#     for i in range(5) :
#         state = env.reset()
#         init_z= env.sim.data.qpos[1]
#         done = False
#         reward = 0
#         episode_reward = 0
#         while not done:
#             state = torch.from_numpy(state).float()
#             action = online_agent.actor_network.get_action(state.to(device))
#             next_state, reward, done, info = env.step(action.detach().cpu().numpy())


#             jump_reward = -(abs(info["x_velocity"]))-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
#             forward_reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
#             backward_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))

#             ## Corresponding Reward Function
#             if args.task == "jump" :
#                 reward = jump_reward
#             elif args.task == "run-forward" :
#                 reward = forward_reward
#             elif args.task == "run-backward" :
#                 reward = backward_reward

#             ## Buffer for Tranining
#             saving_buffer.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([jump_reward]),torch.FloatTensor([forward_reward]),torch.FloatTensor([backward_reward]),\
#                                torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))

#             episode_reward += reward
#             state = next_state
#         print(episode_reward)

# print("Saving Data : ",saving_buffer.size())
# saving_buffer.save_data(str(args.task)+"_"+str(args.data_type))
saving_buffer.save_data(str(args.task)+"_"+str(args.data_type))

print("End Training Online Agent")

