import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import Buffer
import torch.optim as optim
import torch.nn as nn



class CRR(nn.Module):
    def __init__(self,state_dim,hidden,action_dim):
        super(CRR, self).__init__()

        self.actor_network = models.actor(state_dim,hidden,action_dim)
        self.actor_target_network = models.actor(state_dim, hidden, action_dim)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_network = models.critic(state_dim+action_dim,hidden)
        self.critic_target_network = models.critic(state_dim+action_dim,hidden)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_lr = 0.003
        self.critic_lr = 0.003
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(),self.critic_lr)
        self.target_update_period = 100
        # Action Sample Num is using for obtain V(s)
        self.action_sample_num = 10
        self.beta = 1
        self.epoch = 1
        self.discount_factor = 0.99
        self.batch_size = 100
        self.memory = Buffer.Replay_buffer(self.batch_size)


    def load_dict(self):
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

    def train_net(self):
        samples = self.memory.sample()
        state, action, reward, next_state, done = zip(*samples)
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        done = torch.stack(done)


        # Critic Update
        q_val = self.critic_network(state,action)
        with torch.no_grad():
            action_dist = self.actor_target_network(next_state)
            next_action = action_dist.sample()
            q1_val = self.critic_target_network(next_state,next_action)
        critic_loss = F.mse_loss(q_val, reward + q1_val * (1-done))
        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Actor Update
        action_dist = self.actor_network(state)
        log_prob_action = action_dist.log_prob(action).sum(-1,keepdim=True)
        state_value = self.get_estimate_value(state,action)
        advantage = q_val.detach() - state_value
        exp_adv = torch.clamp(torch.exp(advantage),max = 20)
        actor_loss = -(log_prob_action * exp_adv).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return critic_loss,actor_loss


    def get_estimate_value(self,state,actions):
        with torch.no_grad():
            action = self.actor_network(state)
            actions = [action.sample() for _ in range(self.action_sample_num)]
            total_q_val = []

            for act in actions :

                q_pred = self.critic_network(state,act)
                q_pred = q_pred.mean(0)
                total_q_val.append(q_pred)

            value = torch.stack(total_q_val,dim =0).mean(0)
        return value

    def train_critic(self):
            ## TODO : Let us beign!
        raise NotImplementedError

    def train_actor(self):
        ## TODO
        raise NotImplementedError