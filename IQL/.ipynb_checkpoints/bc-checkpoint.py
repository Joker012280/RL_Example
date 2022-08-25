import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn



class BC(nn.Module):
    def __init__(self,state_dim,hidden,action_dim):
        super(BC, self).__init__()

        self.actor_network = models.actor(state_dim,hidden,action_dim)
        self.actor_target_network = models.actor(state_dim, hidden, action_dim)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.actor_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),self.actor_lr)
        self.target_update_period = 100
        self.batch_size = 256
        self.memory = Buffer.Replay_buffer(self.batch_size)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def load_dict(self):
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

    def train_net(self):
        samples = self.memory.sample()
        state, action, _,reward, _,next_state, done = zip(*samples)
        state = torch.cat(state)
        state = state.reshape(-1, self.state_dim)
        next_state = torch.cat(next_state)
        next_state = next_state.reshape(-1, self.state_dim)
        action = torch.cat(action)
        action = action.reshape(-1,self.action_dim)
        reward = torch.cat(reward)
        done = torch.cat(done)


        # Actor Update
        _,action_dist = self.actor_network.evaluate(state)
        log_prob_action = action_dist.log_prob(action).sum(-1,keepdim=True)
        actor_loss = -(log_prob_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def train_critic(self):
            ## TODO : Let us beign!
        raise NotImplementedError

    def train_actor(self):
        ## TODO
        raise NotImplementedError