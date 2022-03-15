import torch
import td3models as models
import torch.optim as optim
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Buffer


class Noisegenerator :
    def __init__(self,mean,sigma):
        self.mean = mean
        self.sigma = sigma
    def generate(self) :
        return np.random.normal(self.mean,self.sigma,1)[0]


class TD3(nn.Module):

    def __init__(self,state_dim,hidden,action_dim):
        super(TD3, self).__init__()


        self.discount_factor = 0.99
        self.tau = 0.005
        self.batch_size = 100
        self.memory = Buffer.Replay_buffer(self.batch_size)

        ## Twin Q network
        self.critic1_network = models.critic(state_dim+action_dim, hidden)
        self.critic2_network = models.critic(state_dim+action_dim, hidden)
        self.critic_learning_rate = 0.003
        ## Optimize 할 때 두개의 네트워크를 같이함.
        self.critic_network_optimizer = optim.Adam(list(self.critic1_network.parameters()) \
                                                   + list(self.critic2_network.parameters()), lr=self.critic_learning_rate)

        ## Actor Network 설정
        self.actor_network = models.actor(state_dim,hidden,action_dim)
        self.actor_learning_rate = 0.003
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_learning_rate)

        ## Target Network를 생성
        ## DeepCopy를 이용함. 서로 영향을 안줌

        self.critic1_target_network = copy.deepcopy(self.critic1_network)
        self.critic2_target_network = copy.deepcopy(self.critic2_network)
        self.actor_target_network = copy.deepcopy(self.actor_network)

        self.noise_generator = Noisegenerator(0, 0.2)

        ## TD3 는 Delayed 라는 특성을 가지고 있음
        self.train_num = 1
        self.delay = 2

    def train_net(self):
        samples = self.memory.sample()
        state, action, reward, next_state, done = zip(*samples)
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        done = torch.stack(done)

        ## Cliping 과 noise를 추가함. / Exploration 효과
        ## Pseudo Code 12,13번 참고
        noisy_action = self.actor_target_network(next_state) + torch.tensor(
            np.clip(self.noise_generator.generate(), -0.5, 0.5))
        noisy_action = torch.clamp(noisy_action, -2, 2).detach()

        ## Twin 이기 때문에 2개의 네트워크에서 값을 가져옴
        backup_value = reward + self.discount_factor * torch.min(self.critic1_target_network(next_state, noisy_action), \
                                                            self.critic2_target_network(next_state, noisy_action)) * (1-done)

        ## 두개의 네트워크를 이용해 MSBE loss 구함
        ## Pseudo Code 14번 참고
        q_loss = F.mse_loss(backup_value.detach(), self.critic1_network(state, action)) \
                 + F.mse_loss(backup_value.detach(), self.critic2_network(state, action))

        ## Optimizer (Critic Network)
        self.critic_network_optimizer.zero_grad()
        q_loss.backward()
        self.critic_network_optimizer.step()

        ## Delay (Optimize 하는 시간이 즉각적으로 이루어지지않음)
        ## Pseudo Code 15,16번 참고
        if self.train_num % self.delay == 0:
            ## Optimizer (Actor Network)
            actor_loss = -self.critic1_network(state, self.actor_network(state)).mean()
            self.actor_network_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_network_optimizer.step()

            ## Parameter Copy
            ## Pseudo code 17번 참고 / Polyak Average
            for param, target_param in zip(self.critic1_network.parameters(), self.critic1_target_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2_network.parameters(), self.critic2_target_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_network.parameters(), self.actor_target_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_num += 1