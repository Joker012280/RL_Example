import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn
import random
import math

class DQN(nn.Module):
    def __init__(self, state_dim, hidden, action_dim,batch_size=None):
        super(DQN, self).__init__()
        self.actor_network = models.actor_discrete(state_dim, hidden,action_dim)
        self.actor_target_network = models.actor_discrete(state_dim, hidden,action_dim)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.action_dim = action_dim
        self.eps_start = 0.99
        self.eps_end = 0.01
        self.eps_decay = 15000
        self.actor_lr = 0.01
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.epoch = 1
        self.tau = 0.05
        self.discount_factor = 0.99
        self.batch_size = 128 if batch_size is None else batch_size
        self.capacity = None
        self.memory = Buffer.Replay_buffer(self.batch_size,)
        self.steps_done = 0

    def slow_load_dict(self):
        for param, target_param in zip(self.actor_network.parameters(), self.actor_target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_load_dict(self):
        for param, target_param in zip(self.actor_network.parameters(), self.actor_target_network.parameters()):
            target_param.data.copy_(param.data)

    def select_det_action(self,state):

        with torch.no_grad():

            action = self.actor_network(state)
        return torch.FloatTensor(action).data.max(1)[1].view(1, 1)

    def select_action(self,state):

        sample = random.random()

        self.steps_done += 1
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * (self.steps_done) / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.actor_network(state)
                return torch.FloatTensor(action).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(self.action_dim)]])

    def train_net(self):
        samples = self.memory.sample()
        state, action, reward, next_state, done = zip(*samples)

        state = torch.cat(state)
        next_state = torch.cat(next_state)
        action = torch.cat(action)
        reward = torch.cat(reward)
        done = torch.cat(done)
        state_action_values = self.actor_network(state).gather(1, action.long())
        next_state_action_values = self.actor_network(next_state)

        next_q_values = self.actor_target_network(next_state)

        next_state_action_values = next_q_values.gather(1,torch.max(next_state_action_values, 1)[1].unsqueeze(1)).squeeze(1)

        expected_state_action_values = reward + (self.discount_factor * next_state_action_values) * (1 - done)

        loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()


        return loss
