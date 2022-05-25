import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn


class IQL(nn.Module):
    def __init__(self, state_dim, hidden, action_dim, tau=None, expectile=None, temperature=None, batch_size=None,
                 is_discrete=False):
        super(IQL, self).__init__()
        self.is_discrete = is_discrete
        if self.is_discrete:
            self.actor_network = models.actor_discrete(state_dim, hidden, action_dim)
            self.critic_network_1 = models.critic_discrete(state_dim, hidden,action_dim)
            self.critic_target_network_1 = models.critic_discrete(state_dim, hidden,action_dim)
            self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
            self.critic_network_2 = models.critic_discrete(state_dim, hidden,action_dim, seed=2)
            self.critic_target_network_2 = models.critic_discrete(state_dim, hidden,action_dim)
            self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
            self.value_network = models.value(state_dim, hidden)
        else:
            self.actor_network = models.actor(state_dim, hidden, action_dim)
            self.critic_network_1 = models.critic(state_dim + action_dim, hidden)
            self.critic_target_network_1 = models.critic(state_dim + action_dim, hidden)
            self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
            self.critic_network_2 = models.critic(state_dim + action_dim, hidden, seed=2)
            self.critic_target_network_2 = models.critic(state_dim + action_dim, hidden)
            self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
            self.value_network = models.value(state_dim, hidden)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_lr = 0.003
        self.critic_lr = 0.003
        self.value_lr = 0.003
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), self.value_lr)
        self.tau = 0.005 if tau is None else tau
        self.expectile = 0.8 if expectile is None else expectile
        self.temperature = 3.0 if temperature is None else temperature
        self.epoch = 1
        self.discount_factor = 0.99
        self.batch_size = 128 if batch_size is None else batch_size
        self.memory = Buffer.Replay_buffer(self.batch_size)

    def slow_load_dict(self):
        for param, target_param in zip(self.critic_network_1.parameters(), self.critic_target_network_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_network_2.parameters(), self.critic_target_network_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_load_dict(self):
        for param, target_param in zip(self.critic_network_1.parameters(), self.critic_target_network_1.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(self.critic_network_2.parameters(), self.critic_target_network_2.parameters()):
            target_param.data.copy_(param.data)

    def get_value_loss(self, q, value):
        weight = torch.where((q - value) > 0, self.expectile, (1 - self.expectile))
        return weight * ((q - value) ** 2)

    def train_net(self):
        samples = self.memory.sample()
        state, action, reward, next_state, done = zip(*samples)
        state = torch.cat(state)
        state = state.reshape(-1, self.state_dim)
        next_state = torch.cat(next_state)
        next_state = next_state.reshape(-1, self.state_dim)
        action = torch.cat(action)
        reward = torch.cat(reward)
        done = torch.cat(done)

        if self.is_discrete:
            with torch.no_grad():
                q_1 = self.critic_target_network_1(state).gather(1, action.long())
                q_2 = self.critic_target_network_2(state).gather(1, action.long())
                min_q = torch.min(q_1, q_2)
            value = self.value_network(state)
            value_loss = self.get_value_loss(min_q, value).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Actor Update
            with torch.no_grad():
                q_1 = self.critic_target_network_1(state).gather(1, action.long())
                q_2 = self.critic_target_network_2(state).gather(1, action.long())
                min_q = torch.min(q_1, q_2)
                value = self.value_network(state)
            advantage = min_q - value
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            exp_adv = torch.clamp(torch.exp((advantage) * self.temperature), max=100)
            _, action_dist = self.actor_network.evaluate(state)
            log_prob_action = action_dist.log_prob(action)
            actor_loss = -(log_prob_action * exp_adv).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic Update
            q_1 = self.critic_network_1(state).gather(1, action.long()).squeeze()
            q_2 = self.critic_network_2(state).gather(1, action.long()).squeeze()
            with torch.no_grad():
                next_value = self.value_network(next_state).squeeze()
                q_target = reward + (self.discount_factor * next_value * (1 - done))
            critic_loss_1 = F.mse_loss(q_1, q_target)
            critic_loss_2 = F.mse_loss(q_2, q_target)
            critic_loss_1 = critic_loss_1.mean()
            critic_loss_2 = critic_loss_2.mean()

            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            self.hard_load_dict()
        else:
            # Value Update
            with torch.no_grad():
                q_1 = self.critic_target_network_1(state, action)
                q_2 = self.critic_target_network_2(state, action)
                min_q = torch.min(q_1, q_2)
            value = self.value_network(state)
            value_loss = self.get_value_loss(min_q, value).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Actor Update
            with torch.no_grad():
                q_1 = self.critic_target_network_1(state, action)
                q_2 = self.critic_target_network_2(state, action)
                min_q = torch.min(q_1, q_2)
                value = self.value_network(state)
            advantage = min_q - value
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            exp_adv = torch.clamp(torch.exp((advantage) * self.temperature), max=100)
            _, action_dist = self.actor_network.evaluate(state)
            log_prob_action = action_dist.log_prob(action)
            actor_loss = -(log_prob_action * exp_adv).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic Update
            q_1 = self.critic_network_1(state, action).squeeze()
            q_2 = self.critic_network_2(state, action).squeeze()
            with torch.no_grad():
                next_value = self.value_network(next_state).squeeze()
                q_target = reward + (self.discount_factor * next_value * (1 - done))
            critic_loss_1 = F.mse_loss(q_1, q_target)
            critic_loss_2 = F.mse_loss(q_2, q_target)
            critic_loss_1 = critic_loss_1.mean()
            critic_loss_2 = critic_loss_2.mean()

            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            self.slow_load_dict()

        return critic_loss_1, critic_loss_2, actor_loss, value_loss

    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError