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


class Idea2(nn.Module):
    def __init__(self,
                 state_dim,
                 hidden,
                 action_dim,
                 tau=None,
                 batch_size = 256,
                 dataloader = None,
                 action_bounds=None,
                 device=None):
        super(Idea2, self).__init__()
        
        self.behavior_train = True
        self.device = device
        self.actor_network = models.actor(state_dim, hidden, action_dim,action_bounds).to(device)
        self.behavior_network = models.behavior(state_dim, hidden, action_dim,action_bounds).to(device)
        self.critic_network_1 = models.critic(state_dim + action_dim, hidden).to(device)
        self.critic_target_network_1 = models.critic(state_dim + action_dim, hidden).to(device)
        self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.critic_network_2 = models.critic(state_dim + action_dim, hidden).to(device)
        assert self.critic_network_1.parameters() != self.critic_network_2.parameters()
        self.critic_target_network_2 = models.critic(state_dim + action_dim, hidden).to(device)
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.behavior_optimizer = optim.Adam(self.behavior_network.parameters(), self.actor_lr, weight_decay = 1e-5)
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.reward_scale = 5.0
        self.log_alpha = torch.tensor([0.0],requires_grad = True)
        self.alpha = self.log_alpha.exp().detach().to(device)
        self.tau = 0.005 if tau is None else tau
        self.dataloader = dataloader
        self.epoch = 1
        self.discount_factor = 0.99
        self.batch_size = batch_size
        
    
    def load_dict(self,critic,critic_target):
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    
    def train_net(self):
        if self.dataloader is None : 
                raise e
        else : 
            samples = next(iter(self.dataloader))
            state,action,reward,next_state,done = samples
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
        
              
        ## Behavior Update
        if self.behavior_train :
            _,action_dist = self.behavior_network.evaluate(state)
            log_prob_action = action_dist.log_prob(action).sum(-1,keepdim=True)
            behavior_loss = -(log_prob_action).mean()
            self.behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.behavior_optimizer.step()
        else :
            behavior_loss = 0.0
        
        with torch.no_grad():
            # Actor Q Target
            next_action,next_log_prob_action = self.actor_network.evaluate(next_state)
            actor_q_target_1 = self.critic_target_network_1(next_state, next_action)
            actor_q_target_2 = self.critic_target_network_2(next_state, next_action)
            actor_q_target = torch.min(actor_q_target_1,actor_q_target_2)
            actor_q_target = actor_q_target - self.alpha * next_log_prob_action
            # Behavior Q Target
            behavior_next_action, behavior_action_dist = self.behavior_network.evaluate(next_state)
            behavior_q_target_1 = self.critic_target_network_1(next_state, behavior_next_action)
            behavior_q_target_2 = self.critic_target_network_2(next_state, behavior_next_action)
            behavior_q_target = torch.min(behavior_q_target_1,behavior_q_target_2)
            behavior_log_prob_action = (behavior_action_dist.log_prob(behavior_next_action)-torch.log(1 - behavior_next_action.pow(2) + 1e-6)).sum(1,keepdim=True)
            behavior_q_target = behavior_q_target - self.alpha * behavior_log_prob_action
            
            ## Behavior Action Prob
            alpha = behavior_action_dist.log_prob(next_action).exp().mean(-1,keepdim=True)
            next_q_target = alpha * actor_q_target + (1-alpha) * behavior_q_target
            
            next_q_target = actor_q_target
            
            q_target = self.reward_scale * reward + (self.discount_factor * next_q_target * (1 - done))

        q1_val = self.critic_network_1(state, action)
        q2_val = self.critic_network_2(state, action)

        critic_loss_1 = F.mse_loss(q1_val,q_target)
        critic_loss_2 = F.mse_loss(q2_val,q_target)
    
    
        ## Loss Weighting by alpha
        
        
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()
        
        
        # Actor Update
        action_pred,log_prob_action = self.actor_network.evaluate(state)
        q1_val = self.critic_network_1(state, action_pred)
        q2_val = self.critic_network_2(state, action_pred)
        q_val = torch.min(q1_val, q2_val)
        actor_loss = ((self.alpha * log_prob_action - q_val) * alpha).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        self.load_dict(self.critic_network_1,self.critic_target_network_1)
        self.load_dict(self.critic_network_2,self.critic_target_network_2)

        return torch.min(critic_loss_1,critic_loss_2), actor_loss, behavior_loss , alpha.mean()

    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError