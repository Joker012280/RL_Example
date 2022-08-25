import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn



class BC(nn.Module):
    def __init__(self,state_dim,hidden,action_dim,action_bounds,dataloader=None,device = None):
        super(BC, self).__init__()

        self.actor_network = models.behavior(state_dim,hidden,action_dim,action_bounds).to(device)
        self.actor_target_network = models.behavior(state_dim, hidden, action_dim,action_bounds).to(device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.actor_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),self.actor_lr,weight_decay = 1e-5)
        self.target_update_period = 100
        self.batch_size = 256
        self.memory = Buffer.Replay_buffer(self.batch_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dataloader = dataloader
        self.device = device

    def load_dict(self):
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

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