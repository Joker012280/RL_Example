import numpy
import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np

class Idea(nn.Module):
    def __init__(self, 
                 state_dim, 
                 hidden, 
                 action_dim,
                 action_bounds,
                 expectile = None,
                 temperature = None, 
                 batch_size = 64,
                 ensemble_num = 3,
                 task_idx = 0,
                 tau = None,
                 dataloader = None,
                 device = None,
                 ):
        super(Idea, self).__init__()
        

        self.task_idx = task_idx
        self.ensemble_num = ensemble_num    
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.value_lr = 3e-4
        self.dataloader = dataloader
        ## Ensemble
        self.critic_networks,self.critic_target_networks,self.critic_optims = [], [], []
        for i in range(self.ensemble_num) :
            critic = models.critic(state_dim + action_dim, hidden).to(device)
            critic_target = models.critic(state_dim + action_dim, hidden).to(device)
            critic_target.load_state_dict(critic.state_dict())
            self.critic_networks.append(critic)
            self.critic_target_networks.append(critic_target)
            self.critic_optims.append(optim.Adam(critic.parameters(), self.critic_lr))        
        
        self.value_network = models.value(state_dim, hidden).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), self.value_lr)
        self.actor_network = models.actor_dist(state_dim, hidden, action_dim,action_bounds).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        
        self.device = device
        # IQL
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = 0.005 if tau is None else tau
        self.temperature = 3.0 if temperature is None else temperature
        self.expectile = expectile
        self.batch_size = batch_size
        self.discount_factor = 0.99
        self.val_weight = None
        
        self.memory = Buffer.Replay_buffer(self.batch_size)
        
    def expectile_cal(self,var):
        # TODO : Think more
        if self.val_weight is None :
            self.val_weight = var
        else :
            self.val_weight = 0.995 * self.val_weight + 0.005 * var
            # self.val_weight = 0.99 * self.val_weight + 0.01 * var
        
    
    def load_dict(self,critic,critic_target):
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_value_loss(self, q, value,var):
        if self.expectile is not None :
            weight = torch.where((q - value) > 0, self.expectile, (1 - self.expectile))
            return weight * ((q - value) ** 2) ,self.expectile
    
        else : 
            self.expectile_cal(var)
            expectile = torch.sigmoid((self.val_weight / var))
            expectile = torch.clamp(expectile,max = 0.9,min=0.5)
            weight = torch.where((q-value) > 0, expectile, (1-expectile))
            return weight * ((q - value) ** 2) ,expectile.mean().item()
    
    def cat_sample(self,*samples) :
        
        state, action, jump_reward,forward_reward,backward_reward, next_state, done = zip(*samples)
        state = torch.cat(state)
        state = state.reshape(-1, self.state_dim).to(self.device)
        next_state = torch.cat(next_state)
        next_state = next_state.reshape(-1, self.state_dim).to(self.device)
        action = torch.cat(action)
        action = action.reshape(-1,self.action_dim).to(self.device)
        jump_reward = torch.cat(jump_reward)
        jump_reward = jump_reward.unsqueeze(1).to(self.device)
        forward_reward = torch.cat(forward_reward)
        forward_reward = forward_reward.unsqueeze(1).to(self.device)
        backward_reward = torch.cat(backward_reward)
        backward_reward = backward_reward.unsqueeze(1).to(self.device)
        done = torch.cat(done)
        done = done.unsqueeze(1).to(self.device)
        
        return state, action, jump_reward,forward_reward,backward_reward, next_state, done

    
    def train_net(self):
        if self.dataloader is None : 
            samples = self.memory.sample()
            state,action,jump_reward,forward_reward,backward_reward,next_state,done = self.cat_sample(*samples)
            if self.task_idx == 0 :
                reward = forward_reward
            elif self.task_idx == 1 :
                reward = backward_reward
            elif self.task_idx == 2 :
                reward = jump_reward
            else :
                raise e
        else : 
            samples = next(iter(self.dataloader))
            state,action,reward,next_state,done = samples
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
        
        if self.ensemble_num > 2 :
            ## Get Variance Only for Q val use it in Value function Learning
            stacked_q_target_vals = torch.stack([self.critic_target_networks[i](state,action).detach() for i in range(self.ensemble_num)])
            ## Mean and Variance
            q_vals_mean = stacked_q_target_vals.mean(0).to(self.device)
            q_vals_var = stacked_q_target_vals.var(0).to(self.device)
            # q_vals_min = stacked_q_target_vals.min(0).to(self.device)
        else : 
            raise e
        
        value = self.value_network(state)
        value_loss,expectile = self.get_value_loss(q_vals_mean, value,q_vals_var)
        value_loss = value_loss.mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Actor Update
        with torch.no_grad():
            value = self.value_network(state)
        advantage = q_vals_mean - value
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        exp_adv = torch.clamp(torch.exp((advantage) * self.temperature), max=100)
        _, _,action_dist = self.actor_network.evaluate(state)
        log_prob_action = action_dist.log_prob(action)
        actor_loss = -(log_prob_action * exp_adv).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss_list = []
        with torch.no_grad():
            next_value = self.value_network(next_state)
        for i in range(self.ensemble_num) :
            q_val = self.critic_networks[i](state,action)
            q_target = reward + (self.discount_factor * next_value * (1 - done))
            critic_loss = F.mse_loss(q_val,q_target).mean()
            self.critic_optims[i].zero_grad()
            critic_loss.backward()
            self.critic_optims[i].step()
            critic_loss_list.append(critic_loss.detach().cpu())
        
        for i in range(self.ensemble_num) :
            self.load_dict(self.critic_networks[i],self.critic_target_networks[i])
        
        return np.mean(critic_loss_list), value_loss.item() , actor_loss.item() , q_vals_var.detach().cpu().numpy(),expectile
    
    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError