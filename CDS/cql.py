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


class CQL(nn.Module):
    def __init__(self, state_dim, hidden, action_dim, action_bounds,dataloader = None,tau=None, target_entropy=None, temperature=None,batch_size = 256,reward_scale=5,with_lagrange = True,device=None):
        super(CQL, self).__init__()
        
        self.device = device
        self.actor_network = models.actor(state_dim, hidden, action_dim,action_bounds).to(device)
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
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        
        self.dataloader = dataloader
        
        # SAC & CQL
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_lr = 3e-4
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)
        self.temperature = 1.0
        self.reward_scale = reward_scale
        self.tau = 0.005 if tau is None else tau
        
        self.cql_weight = 1.0
        self.with_lagrange = with_lagrange
        self.target_action_gap = 5.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=self.critic_lr) 
        self.discount_factor = torch.FloatTensor([0.99]).to(device)
        self.clip_parameter = 1
        self.batch_size = batch_size
        self.memory = Buffer.Replay_buffer(batch_size = self.batch_size,capacity = 1e+6)

    def load_dict(self):
        for param, target_param in zip(self.critic_network_1.parameters(), self.critic_target_network_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_network_2.parameters(), self.critic_target_network_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _compute_policy_values(self, obs_pi, obs_q):
        with torch.no_grad():
            action,log_action_prob = self.actor_network.evaluate(obs_pi)
        q1_val = self.critic_network_1(obs_q, action)
        q2_val = self.critic_network_2(obs_q, action)

        return q1_val - log_action_prob, q2_val - log_action_prob

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_dim)
        return random_values - random_log_probs
    
    def train_net(self):
        if self.dataloader is None : 
            samples = self.memory.sample()

            state, action, jump_reward,reward,backward_reward, next_state, done = zip(*samples)

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
        else : 
            samples = next(iter(self.dataloader))
            state,action,reward,next_state,done = samples
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
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
            q_target = self.reward_scale*reward + (self.discount_factor * next_q_target * (1 - done))

        q1_val = self.critic_network_1(state, action)
        q2_val = self.critic_network_2(state, action)

        critic_loss_1 = F.mse_loss(q1_val, q_target)
        critic_loss_2 = F.mse_loss(q2_val, q_target)

#             self.critic_optimizer_1.zero_grad()
#             critic_loss_1.backward()
#             self.critic_optimizer_1.step()

#             self.critic_optimizer_2.zero_grad()
#             critic_loss_2.backward()
#             self.critic_optimizer_2.step()


        random_action = torch.FloatTensor(q1_val.shape[0] * 10, action.shape[-1]).uniform_(-1, 1).to(self.device)
        number_repeat = int(random_action.shape[0] / state.shape[0])
        temp_state = state.unsqueeze(1).repeat(1, number_repeat, 1).view(state.shape[0] * number_repeat, state.shape[1])
        temp_next_state = next_state.unsqueeze(1).repeat(1, number_repeat, 1).view(next_state.shape[0] * number_repeat,
                                                                                   next_state.shape[1])

        current_pi_value_1, current_pi_value_2 = self._compute_policy_values(temp_state, temp_state)
        next_pi_value_1, next_pi_value_2 = self._compute_policy_values(temp_next_state, temp_state)

        random_value_1 = self._compute_random_values(temp_state, random_action, self.critic_network_1).reshape(
            state.shape[0],
            number_repeat, 1)
        random_value_2 = self._compute_random_values(temp_state, random_action, self.critic_network_2).reshape(
            state.shape[0],
            number_repeat, 1)

        current_pi_value_1 = current_pi_value_1.reshape(state.shape[0], number_repeat, 1)
        current_pi_value_2 = current_pi_value_2.reshape(state.shape[0], number_repeat, 1)

        next_pi_value_1 = next_pi_value_1.reshape(state.shape[0], number_repeat, 1)
        next_pi_value_2 = next_pi_value_2.reshape(state.shape[0], number_repeat, 1)

        cat_q_1 = torch.cat([random_value_1, current_pi_value_1, next_pi_value_1], 1)
        cat_q_2 = torch.cat([random_value_2, current_pi_value_2, next_pi_value_2], 1)

        assert cat_q_1.shape == (state.shape[0], 3 * number_repeat, 1), f"cat_q1 instead has shape: {cat_q_1.shape}"
        assert cat_q_2.shape == (state.shape[0], 3 * number_repeat, 1), f"cat_q1 instead has shape: {cat_q_2.shape}"

        cql_scaled_loss_1 = ((torch.logsumexp(cat_q_1 / self.temperature,
                                              dim=1).mean() * self.cql_weight * self.temperature) - q1_val.mean()) * self.cql_weight
        cql_scaled_loss_2 = ((torch.logsumexp(cat_q_2 / self.temperature,
                                              dim=1).mean() * self.cql_weight * self.temperature) - q2_val.mean()) * self.cql_weight
        
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])

        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql_scaled_loss_1 = cql_alpha * (cql_scaled_loss_1 - self.target_action_gap)
            cql_scaled_loss_2 = cql_alpha * (cql_scaled_loss_2 - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql_scaled_loss_1 - cql_scaled_loss_2) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        
        
        
        
        total_critic_loss_1 = critic_loss_1 + cql_scaled_loss_1
        total_critic_loss_2 = critic_loss_2 + cql_scaled_loss_2

        self.critic_optimizer_1.zero_grad()
        total_critic_loss_1.backward(retain_graph=True)
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        total_critic_loss_2.backward()
        self.critic_optimizer_2.step()

        self.load_dict()

        return total_critic_loss_1.item(), total_critic_loss_2.item(), actor_loss

    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError