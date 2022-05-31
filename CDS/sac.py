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


class SAC(nn.Module):
    def __init__(self, state_dim, hidden, action_dim, tau=None, target_entropy=None, temperature=None,batch_size = 256,device=None):
        super(SAC, self).__init__()
        
        self.device = device
        self.actor_network = models.actor(state_dim, hidden, action_dim).to(device)
        self.critic_network_1 = models.critic(state_dim + action_dim, hidden,seed = 1).to(device)
        self.critic_target_network_1 = models.critic(state_dim + action_dim, hidden).to(device)
        self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.critic_network_2 = models.critic(state_dim + action_dim, hidden,seed =2 ).to(device)
        assert self.critic_network_1.parameters() != self.critic_network_2.parameters()
        self.critic_target_network_2 = models.critic(state_dim + action_dim, hidden).to(device)
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
        self.actor_lr = 1e-4
        self.critic_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        # SAC & CQL
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_lr = 3e-4
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)
        self.temperature = 1.0

        self.tau = 0.005 if tau is None else tau
        
        self.epoch = 1
        self.discount_factor = torch.FloatTensor([0.99]).to(device)
        self.clip_parameter = 1
        self.batch_size = batch_size
        self.memory = Buffer.Replay_buffer(self.batch_size)

    def load_dict(self):
        for param, target_param in zip(self.critic_network_1.parameters(), self.critic_target_network_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_network_2.parameters(), self.critic_target_network_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_net(self):
        for i in range(self.epoch):
            samples = self.memory.sample()
            state, action, reward, next_state, done = zip(*samples)
            state = torch.cat(state)
            state = state.reshape(-1, self.state_dim).to(self.device)
            next_state = torch.cat(next_state)
            next_state = next_state.reshape(-1, self.state_dim).to(self.device)
            action = torch.cat(action)
            action = action.reshape(-1,self.action_dim).to(self.device)
            reward = torch.cat(reward)
            reward = reward.unsqueeze(1).to(self.device)
            done = torch.cat(done)
            done = done.unsqueeze(1).to(self.device)

            # Actor Update
            current_alpha = copy.deepcopy(self.alpha)
            action_pred,log_prob_action = self.actor_network.evaluate(state)
            q1_val = self.critic_network_1(state, action_pred)
            q2_val = self.critic_network_2(state, action_pred)
            q_val = torch.min(q1_val, q2_val).cpu()
            actor_loss = (current_alpha * log_prob_action.cpu() - q_val).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # SAC Alpha loss
            alpha_loss = - (self.log_alpha.exp() * (log_prob_action.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            # SAC Critic Update
            with torch.no_grad():
                next_action,next_log_prob_action = self.actor_network.evaluate(next_state)
                next_q_target_1 = self.critic_target_network_1(next_state, next_action)
                next_q_target_2 = self.critic_target_network_2(next_state, next_action)
                next_q_target = torch.min(next_q_target_1, next_q_target_2)
                next_q_target = next_q_target - self.alpha.to(self.device) * next_log_prob_action
                q_target = reward + (self.discount_factor * next_q_target * (1 - done))

            q1_val = self.critic_network_1(state, action)
            q2_val = self.critic_network_2(state, action)

            critic_loss_1 = F.mse_loss(q1_val, q_target)
            critic_loss_2 = F.mse_loss(q2_val, q_target)

            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # CQL # TODO : Understand CQL!

            self.load_dict()

        return critic_loss_1, critic_loss_2, actor_loss

    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError