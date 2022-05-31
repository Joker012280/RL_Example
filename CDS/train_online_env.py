import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cds
import sac
import os
import sys
import mujoco_py
import Buffer
import argparse

def plot_durations(name):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(list_total_reward)
    plt.title('Training')
    plt.xlabel('num of episode')
    plt.ylabel('reward')
    plt.plot(durations_t.numpy())
    plt.savefig(name)


parser = argparse.ArgumentParser(description='RL')
parser.add_argument("--env", type=str, default="Walker2d-v3", help="Gym env name, default : Walker2d-v3")
parser.add_argument("--task", type=str, default="run-forward",help="Specific Task, default : run-forward")
parser.add_argument("--episodes", type=int, default=500, help="Number of episodes, default : 500")
parser.add_argument("--data_type", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Env : {} || Task : {} || Num Episode : {} || Device : {} || {} || {} ".format(args.env,args.task,args.episodes,device,torch.cuda.current_device(),torch.cuda.device_count()))

## Task checkning
if args.task == "run-forward" :
    healthy_reward = 1.0
    forward_reward_weight = 1.0
elif args.task == "run-backward" :
    healthy_reward = 1.0
    forward_reward_weight = -1.0
elif args.task == "jump" :
    healthy_reward = 0.0
    forward_reward_weight = -1.0
else :
    raise NotImplementedError
    

## Environment
env = gym.make(args.env, healthy_reward = healthy_reward, forward_reward_weight = forward_reward_weight)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_episode_num = args.episodes
hidden = 256
print_interval = 100
saving_buffer = Buffer.Replay_buffer(256)


## Train
total_reward = 0
online_agent = sac.SAC(state_dim,hidden,action_dim,device=device)
list_total_reward = []
best_reward = float("-inf")

if args.data_type == "expert" :
    max_episode_num = 5
    print_interval = 1
    model_path = "results/SAC_jump_Walker2d_best.pth"
    temp = torch.load(model_path)
    online_agent.load_state_dict(temp['model_state_dict'])
    online_agent.eval()
    print("End Loading")



for num_episode in range(max_episode_num):
    state = env.reset()
    init_z= env.sim.data.qpos[1]
    global_step = 0
    done = False
    reward = 0
    while not done:
        global_step += 1
        state = torch.from_numpy(state).float()
        action = online_agent.actor_network.get_action(state.to(device))


        next_state, reward, done, info = env.step(action.detach().cpu().numpy())
        
        if args.task == "jump" :
            reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
        elif args.task == "run-forward" :
            reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
        elif args.task == "run-backward" :
            reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            
        ## Buffer
        online_agent.memory.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([reward]),\
                           torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
        if args.data_type == "medium-replay" or args.data_type == "medium":
            if num_episode >= 100:
                saving_buffer.push((state.cpu(), action.detach().cpu().unsqueeze(1), torch.FloatTensor([reward]),\
                               torch.FloatTensor(np.array(next_state)), torch.FloatTensor([done])))
    
        state = next_state

        total_reward += reward

        if done:
            break
        ## Memory size가 커지고 나서 학습시작
        if online_agent.memory.size() > 500 :
            if args.data_type == "medium" and num_episode >= 100 :
                pass
            else :
                q1_loss,q2_loss,actor_loss = online_agent.train_net()
     
    ## 결과값 프린트
    if num_episode % print_interval == 0 and num_episode != 0:
        
        print("Num of episode : {} || average score : {:.1f} ".format(num_episode, \
                                                                 total_reward / print_interval))
        list_total_reward.append(total_reward / print_interval)
        if actor_loss is not None:
            print("Policy Loss : {:.1f} || Q1 Loss : {:.1f} || Q2 Loss : {:.1f} ".format(actor_loss,q1_loss,q2_loss))
        if total_reward > best_reward and not args.data_type == "expert":
            torch.save({
                            'model_state_dict': online_agent.state_dict(),
                        }, 'results/SAC_'+str(args.task)+'_Walker2d_best.pth')
            best_reward = total_reward
        
        total_reward = 0.0

plot_durations("results/Online_Agent_"+str(args.task)+".png")
#online_agent.memory.save_data("online")
saving_buffer.save_data(str(args.task)+"_"+str(args.data_type))

print("Finish Data Saving")

## Model Save
if args.data_type != "expert" :
    torch.save({
        'model_state_dict': online_agent.state_dict(),
    }, 'results/SAC_'+str(args.task)+'_Walker2d.pth')

clear_output()

print("End Training Online Agent")

