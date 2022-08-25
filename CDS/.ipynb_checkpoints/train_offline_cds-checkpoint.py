import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import cds
# import cql
import sac
# import sac_q
import os
import sys
import mujoco_py
import Buffer
import argparse
from distutils.util import strtobool




def plot_durations(name,total,average,behavior,task):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(total)
    durations_tt = torch.FloatTensor(average)
    plt.title('Training_'+str(task)+'_'+str(args.weight)+'_'+str(args.data_sharing))
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
parser.add_argument("--algo", type=str,default ="cds")
parser.add_argument("--epoch", type=int, default=1500000, help="Number of epoch, default : 200000")
parser.add_argument("--weight", type=str, default="cds")
parser.add_argument("--with_lagrange",type=strtobool,default= False)
parser.add_argument("--reward_scale", type=float, default=5)
parser.add_argument("--gpu",type=str,default = "0")
parser.add_argument("--task_num",type=int,default = 3)
parser.add_argument("--task_idx",type=int,default=None) # Task 0 : Forward, 1 : Backward, 2 : Jump
parser.add_argument("--data", nargs='+', default=[])
parser.add_argument("--data_sharing", type=strtobool, default=True)

args = parser.parse_args()
cuda = "cuda:"+args.gpu
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    

## Environment
env = gym.make(args.env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = [env.action_space.low[0], env.action_space.high[0]]

max_epoch = args.epoch
batch_size = 128
hidden = 256
print_interval = args.print_interval
weight_temperature = [5,float("inf")]


# Condition Check
if args.task_idx is None and args.task_num == 1 :
    raise e    


## Train
agent = cds.CDS(state_dim,hidden,action_dim,weight_temperature = weight_temperature,weight = args.weight,reward_scale = args.reward_scale,action_bounds = action_bounds,task_num = args.task_num,task_idx = args.task_idx,with_lagrange=args.with_lagrange,data_sharing=args.data_sharing,device=device)    

for i,data_name in enumerate(args.data) :
    agent.memory.load_data(data_name,i)

# agent.memory.load_data("run-forward_medium-replay")
# agent.memory_2.load_data("run-backward_medium")
# agent.memory_3.load_data("jump_expert")


print("Env : {} || Num Epoch : {} || Weight : {} || Device : {} ".format(args.env,args.epoch,args.weight,device))
print("Memory Size : {} || Task Num {} || Data Sharing {}".format(agent.memory.size(),args.task_num,args.data_sharing))



def testing(task,agent,behavior_agent = False):
    ## Task state expansion
    if task == 0 and args.task_num != 1 :
        task_state = np.array([1,0,0])
        task_state = torch.from_numpy(task_state)
        env = gym.make(args.env)
    
    elif task == 1 and args.task_num != 1:
        task_state = np.array([0,1,0])
        task_state = torch.from_numpy(task_state)
        env = gym.make(args.env)
    
    elif task == 2 and args.task_num != 1:
        task_state = np.array([0,0,1])
        task_state = torch.from_numpy(task_state)
        healthy_z_range = (0.8,10)
        env = gym.make(args.env,healthy_z_range = healthy_z_range)
    else :
        task_state = torch.FloatTensor([])
        if args.task_idx == 2 :
            healthy_z_range = (0.8,10)
            env = gym.make(args.env,healthy_z_range = healthy_z_range)
        else :
            env = gym.make(args.env)
    if behavior_agent :
        task_state = task_state = torch.FloatTensor([])
    
    max_episode_num = 5
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
            state = torch.cat([state,task_state]).to(device)
            action = agent.actor_network.get_action(state)
            
            next_state, reward, done, info = env.step(action.cpu().detach().numpy())
            
            ## Task Reward
            jump_reward = -(abs(info["x_velocity"]))-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
            forward_reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            backward_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            
            # Reward Selecting
            if (task == 0 and args.task_num != 1) or (args.task_idx == 0) :
                reward = forward_reward
            elif (task == 1 and args.task_num != 1) or (args.task_idx == 1) :
                reward = backward_reward
            elif (task ==2 and args.task_num != 1) or (args.task_idx == 2) :
                reward = jump_reward
            else :
                raise e
        
            
            state = next_state

            total_reward += reward

            if done:
                break
                
    env.close()
    
    return total_reward / max_episode_num


if args.task_idx is not None : 
    args.task_num = 1

total_reward = [0 for i in range(args.task_num)]
list_total_reward = [[0] for i in range(args.task_num)]
list_average_reward = [[0] for i in range(args.task_num)]
best_reward = [float("-inf") for i in range(args.task_num)]
actor_loss = None

## Training
for epoch in range(1,max_epoch):               
    q1_loss,actor_loss,weight,log_alpha = agent.train_net()
    ## Result Log
    if epoch % print_interval == 0 and epoch != 0:
        for i in range(args.task_num) : 
            average_reward = testing(i,agent)
            if total_reward[i] == 0 :
                total_reward[i] = average_reward
            else :
                total_reward[i] = 0.99 * total_reward[i] + 0.01 * average_reward
            
            print("Epoch : {} || Task : {} || Episode Reward : {:.1f} || Running score : {:.1f} ".format(epoch, i,average_reward, total_reward[i]))
            
            if list_total_reward[i][0] == 0:
                list_total_reward[i][0] = total_reward[i]
            else : 
                list_total_reward[i].append(total_reward[i])
            
            if list_average_reward[i][0] == 0 :
                list_average_reward[i][0] = average_reward
            else :
                list_average_reward[i].append(average_reward)
                                            
            ## Save Best
            if best_reward[i] < average_reward :
                best_reward[i] = average_reward
                torch.save({
                    'model_state_dict': agent.state_dict(),
                }, 'cds_results/'+str(args.weight)+'_'+str(args.data)+'_task_num_'+str(args.task_num)+'_'+str(args.data_sharing)+"_"+str(args.with_lagrange)+'_best_Walker2d.pth')
            
            ## Plot Saving
            if (i == 0 and args.task_num != 1) or (args.task_idx == 0) :
                task_name = "run-forward"
            elif (i == 1 and args.task_num != 1) or (args.task_idx == 1) :
                task_name = "run-backward"
            elif (i == 2 and args.task_num != 1) or (args.task_idx == 2) :
                task_name = "jump"
            else :
                raise e
    
            plot_durations("cds_results/"+str(args.weight)+'_'+task_name+'_'+str(args.data)+'_task_num_'+str(args.task_num)+"_"+str(args.data_sharing)+"_"+str(args.with_lagrange)+".png",list_total_reward[i],list_average_reward[i],0,task_name)
                
                                            
            
        if actor_loss is not None:
            print("Policy Loss : {:.1f} || Q Loss : {:.1f} || Weight : {:.3f} || CQL-log-Alpha : {:.3f} ".format(actor_loss,q1_loss,weight,log_alpha))

            
# Saving Model            
torch.save({
        'model_state_dict': agent.state_dict(),
    }, 'cds_results/'+str(args.algo)+'_'+str(args.data)+'_task_num_'+str(args.task_num)+'_'+str(args.data_sharing)+"_"+str(args.with_lagrange)+'_Walker2d.pth')

for task in range(args.task_num):
    if (task == 0 and args.task_num != 1) or (args.task_idx == 0) :
        task_name = "run-forward"
    elif (task == 1 and args.task_num != 1) or (args.task_idx == 1) :
        task_name = "run-backward"
    elif (task ==2 and args.task_num != 1) or (args.task_idx == 2) :
        task_name = "jump"
    else :
        raise e
    
    ## Testing Original
    behavior_path = "./results/sac_"+str(args.data[task])+"_Walker2d.pth"
    temp = torch.load(behavior_path)
    behavior_agent = sac.SAC(state_dim,hidden,action_dim,action_bounds=action_bounds,reward_scale = args.reward_scale,device=device)
    behavior_agent.load_state_dict(temp['model_state_dict'])
    behavior_agent.eval()
    print("Behavior Loaded")
    behavior_result = testing(task,behavior_agent,True)
    
    
    plot_durations("cds_results/"+str(args.weight)+'_'+task_name+'_'+str(args.data)+'_task_num_'+str(args.task_num)+"_"+str(args.data_sharing)+"_"+str(args.with_lagrange)+".png",list_total_reward[task],list_average_reward[task],behavior_result,task_name)

print("End Training Offline Agent")

