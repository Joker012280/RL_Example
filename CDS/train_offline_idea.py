import gym
import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import Idea
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


def plot_durations(name,total,average,behavior,task):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(total)
    durations_tt = torch.FloatTensor(average)
    plt.title('Training_'+str(task)+'_'+str(args.algo)+'_'+str(args.temperature))
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
parser.add_argument("--algo", type=str,default ="Idea")
parser.add_argument("--epoch", type=int, default=500000, help="Number of epoch, default : 200000")
parser.add_argument("--expectile", type = float , default = None)
parser.add_argument("--temperature",type = float,default = 3.0)
parser.add_argument("--ensemble_num",type=int,default=5)
parser.add_argument("--gpu",type=str,default = "0")
parser.add_argument("--task_idx",type=int,default=0) # Task 0 : Forward, 1 : Backward, 2 : Jump
parser.add_argument("--data", nargs='+', default=[])

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
if len(args.data) == 0 : 
    dataloader = prep_dataloader()
else :
    dataloader = None
## Train
agent = Idea.Idea(state_dim,
                  hidden,
                  action_dim,
                  action_bounds = action_bounds,
                  expectile = args.expectile,
                  temperature = args.temperature,
                  ensemble_num = args.ensemble_num,
                  task_idx = args.task_idx,
                  dataloader = dataloader,
                  device=device)    

for i,data_name in enumerate(args.data) :
    agent.memory.load_data(data_name,i)

# agent.memory.load_data("run-forward_medium-replay")
# agent.memory_2.load_data("run-backward_medium")
# agent.memory_3.load_data("jump_expert")


print("Env : {} || Num Epoch : {} || Device : {} ".format(args.env,args.epoch,device))
print("Memory Size : {} || Task IDX {}".format(agent.memory.size(),args.task_idx))
if len(args.data) == 0 :
    args.data = "d4rl"


def testing(agent,behavior_agent = False):
    ## Task state expansion
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
            state = torch.FloatTensor(state).to(device)
            action = agent.actor_network.get_action(state)
            
            next_state, reward, done, info = env.step(action.cpu().detach().numpy())
            
            ## Task Reward
            jump_reward = -(abs(info["x_velocity"]))-0.001 * np.sum(np.square(action.detach().cpu().numpy())) + 10*(env.sim.data.qpos[1] - init_z)
            forward_reward = +(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            backward_reward = -(info["x_velocity"])-0.001 * np.sum(np.square(action.detach().cpu().numpy()))
            
            # Reward Selecting
            if args.task_idx == 0 :
                reward = forward_reward
            elif args.task_idx == 1 :
                reward = backward_reward
            elif args.task_idx == 2 :
                reward = jump_reward
            else :
                raise e
        
            
            state = next_state

            total_reward += reward

            if done:
                break
                
    env.close()
    
    return total_reward / max_episode_num

if args.task_idx == 0 :
    task_name = "run-forward"
elif args.task_idx == 1 :
    task_name = "run-backward"
elif args.task_idx == 2 :
    task_name = "jump"
else :
    raise e

total_reward = 0
list_total_reward = []
list_average_reward = []
best_reward = float("-inf")
actor_loss = None

## Training
for epoch in range(1,max_epoch):               
    q_loss,value_loss,actor_loss,q_var,var_weight = agent.train_net()
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
            }, 'idea_result/'+str(args.algo)+'_'+str(args.data)+'_'+str(args.ensemble_num)+"_"+str(args.task_idx)+"_"+str(args.expectile)+'_best_Walker2d.pth')

        ## Plot Saving

        plot_durations("idea_result/"+str(args.algo)+'_'+str(args.data)+'_'+
                       str(args.ensemble_num)+"_"+str(args.task_idx)+"_"+str(args.expectile)+".png",list_total_reward,list_average_reward,0,task_name)



        if actor_loss is not None:
            print("Critic Loss : {:.1f} || Value Loss : {:.1f} || Actor Loss : {:.1f} || Variance : {:.1f} || Expectile : {:.3f} ".format(q_loss,value_loss,actor_loss,np.mean(q_var),var_weight))

            
# Saving Model            
torch.save({
        'model_state_dict': agent.state_dict(),
    }, 'idea_result/'+str(args.algo)+'_'+str(args.data)+'_'+str(args.ensemble_num)+"_"+str(args.task_idx)+"_"+str(args.expectile)+'_Walker2d.pth')


    
## Testing Original
if args.data == "d4rl" :
    behavior_result = 2760.3
    plot_durations("idea_result/"+str(args.algo)+'_'+str(args.data)+'_'+str(args.ensemble_num)+"_"+str(args.task_idx)+"_"+str(args.expectile)+".png",list_total_reward,list_average_reward,behavior_result,task_name)

    
else : 
    behavior_path = "./results/sac_"+str(args.data[0])+"_Walker2d.pth"
    temp = torch.load(behavior_path)
    behavior_agent = sac.SAC(state_dim,hidden,action_dim,action_bounds=action_bounds,reward_scale =5.0,device=device)
    behavior_agent.load_state_dict(temp['model_state_dict'])
    behavior_agent.eval()
    print("Behavior Loaded")
    behavior_result = testing(behavior_agent,True)
    plot_durations("idea_result/"+str(args.algo)+'_'+str(args.data)+'_'+str(args.ensemble_num)+"_"+str(args.task_idx)+"_"+str(args.expectile)+".png",list_total_reward,list_average_reward,behavior_result,task_name)

print("End Training Offline Agent")

