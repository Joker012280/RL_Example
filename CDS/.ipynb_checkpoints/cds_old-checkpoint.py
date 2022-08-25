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

class CDS(nn.Module):
    def __init__(self, state_dim, hidden, action_dim,weight_temperature,action_bounds,weight='cdsbasic', reward_scale = 1, tau=None, target_entropy=None, temperature=None , batch_size = 256,task_num =3,device = None,with_lagrange = False):
        super(CDS, self).__init__()

        self.actor_network = models.actor(state_dim + task_num, hidden, action_dim,action_bounds).to(device)
        self.critic_network_1 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_1 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.critic_network_2 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        assert self.critic_network_1.parameters() != self.critic_network_2.parameters()
        self.critic_target_network_2 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
        self.actor_lr = 1e-4
        self.critic_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        
        
 
        self.device = device
        
        # SAC & CQL
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = 1.0
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        # self.alpha_lr = 3e-4
        # self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)
        self.cql_weight = 1.0
        self.temperature = 1.0
        self.reward_scale = reward_scale
        self.target_action_gap = 10.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=self.critic_lr) 
        self.with_lagrange = with_lagrange
        self.tau = 0.005 if tau is None else tau

        # CDS and Task Check
        self.task_num = task_num
        
        self.weight_method = weight
        self.discount_factor = 0.99
        self.weight_temperature = weight_temperature
        self.current_weight_temperature = None
        self.batch_size = batch_size
        
        
        self.memory = Buffer.Replay_buffer(int(self.batch_size/2))
        self.memory_2 = Buffer.Replay_buffer(int(self.batch_size/2))
        self.memory_3 = Buffer.Replay_buffer(int(self.batch_size/2))
    
    
    def load_dict(self):
        for param, target_param in zip(self.critic_network_1.parameters(), self.critic_target_network_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_network_2.parameters(), self.critic_target_network_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def cds_weight_decay(self,relabel_weight_mean):
        self.current_weight_temperature = self.current_weight_temperature * 0.005 + relabel_weight_mean *  0.995
        self.current_weight_temperature = max(min(self.current_weight_temperature,self.weight_temperature[1]),self.weight_temperature[0])
    
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
    
    def get_conservative_qval(self,state,action,next_state) : 
        with torch.no_grad() :
            action_pred,log_prob_action = self.actor_network.evaluate(state)
            q1_val = self.critic_network_1(state, action_pred)
            q2_val = self.critic_network_2(state, action_pred)
        
            
            random_action = torch.FloatTensor(q1_val.shape[0] * 10, action.shape[-1]).uniform_(-1, 1).to(self.device)
            number_repeat = int(random_action.shape[0] / state.shape[0])
            temp_state = state.unsqueeze(1).repeat(1, number_repeat, 1).view(state.shape[0] * number_repeat, state.shape[1])
            temp_next_state = next_state.unsqueeze(1).repeat(1, number_repeat, 1).view(next_state.shape[0] * number_repeat,next_state.shape[1])

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

            cql_q1 = ((torch.logsumexp(cat_q_1,dim=1)) - q1_val)
            cql_q2 = ((torch.logsumexp(cat_q_2,dim=1)) - q2_val)
            
            cql_val = torch.min(cql_q1,cql_q2)
        return cql_val
        
    def calcul_divergence_weight(self,task_state,state,action,next_state) :
        if self.task_ind == 0 :
            samples = self.memory.sample()
        elif self.task_ind == 1 :
            samples = self.memory_2.sample()
        else :
            samples = self.memory_3.sample()
        ori_state, ori_action, _, _, _, ori_next_state, _  = zip(*samples)
        ori_state = torch.cat(ori_state)
        ori_state = ori_state.reshape(-1, self.state_dim).to(self.device)
        ori_state = torch.cat([ori_state,task_state],axis=1).to(self.device)

        ori_next_state = torch.cat(ori_next_state)
        ori_next_state = ori_next_state.reshape(-1, self.state_dim).to(self.device)
        ori_next_state = torch.cat([ori_next_state,task_state],axis=1).to(self.device)

        ori_action = torch.cat(ori_action)
        ori_action = action.reshape(-1,self.action_dim).to(self.device)
        
        ori_conservative_qval = self.get_conservative_qval(ori_state,ori_action,ori_next_state)
        diff_task_conservative_qval = self.get_conservative_qval(state,action,next_state)
        
        delta_term = ori_conservative_qval - diff_task_conservative_qval
        
        if self.current_weight_temperature is None :
            self.current_weight_temperature = delta_term.mean()
        
        weight = torch.sigmoid(delta_term / self.current_weight_temperature)
        self.cds_weight_decay(delta_term.mean())
        return weight

    def calcul_quantile_weight(self,task_state,state,action):
        ## TODO :  Conservative Value
        if self.task_ind == 0 :
            samples = self.memory.sample()
        elif self.task_ind == 1 :
            samples = self.memory_2.sample()
        else :
            samples = self.memory_3.sample()
        ori_state, ori_action, _, _, _, _, _ = zip(*samples)
        ori_state = torch.cat(ori_state)
        ori_state = ori_state.reshape(-1, self.state_dim).to(self.device)
        ori_state = torch.cat([ori_state,task_state],axis=1).to(self.device)

        ori_action = torch.cat(ori_action)
        ori_action = action.reshape(-1,self.action_dim).to(self.device)

        with torch.no_grad() :
            diff_q1_val = self.critic_network_1(state,action)
            diff_q2_val = self.critic_network_2(state,action)
        
            ori_q1_val = self.critic_network_1(ori_state,ori_action)
            ori_q2_val = self.critic_network_2(ori_state,ori_action)
            
            diff_qval = torch.min(diff_q1_val,diff_q2_val)
            ori_qval = torch.min(ori_q1_val,ori_q2_val)
            
            delta_term = diff_qval - ori_qval.mean()
            
            
        if self.current_weight_temperature is None :
            self.current_weight_temperature = delta_term.mean()
        
        weight = torch.sigmoid(delta_term / self.current_weight_temperature)
        self.cds_weight_decay(delta_term.mean())
        return weight
    
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
    
    
    def trunc_transition(self,state,action,reward,next_state,done) :
        
        state,_ = torch.chunk(state,2,dim=0)
        action,_ = torch.chunk(action,2,dim=0)
        reward,_ = torch.chunk(reward,2,dim=0)
        next_state,_ = torch.chunk(next_state,2,dim=0)
        done,_ = torch.chunk(done,2,dim=0)
        
        return state,action,reward,next_state,done
        
    def train_net(self):
        
        # Random Sample Task
        # self.task_ind = random.randint(0,2)
        
        # For all task make update 
        for i in range(self.task_num) :
            self.task_ind = i
        
            if self.task_ind == 0 :
                task_state = np.array([[1,0,0]])
                task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0)).to(self.device)
            elif self.task_ind == 1 :
                task_state = np.array([[0,1,0]])
                task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0)).to(self.device)
            elif self.task_ind == 2 :
                task_state = np.array([[0,0,1]])
                task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0)).to(self.device)


            samples_1 = self.memory.sample()
            samples_2 = self.memory_2.sample()
            samples_3 = self.memory_3.sample()


            state_1, action_1, jump_reward_1,forward_reward_1,backward_reward_1, next_state_1, done_1 = self.cat_sample(*samples_1)
            state_2, action_2, jump_reward_2,forward_reward_2,backward_reward_2, next_state_2, done_2 = self.cat_sample(*samples_2)
            state_3, action_3, jump_reward_3,forward_reward_3,backward_reward_3, next_state_3, done_3 = self.cat_sample(*samples_3)


            ## GET WEIGHT STATE AND REWARD
            if self.task_ind == 0 :

                reward_1 = forward_reward_1
                reward_2 = forward_reward_2
                reward_3 = forward_reward_3

                state_2, action_2, reward_2, next_state_2, done_2 = self.trunc_transition(state_2, action_2, reward_2, next_state_2, done_2)
                state_3, action_3, reward_3, next_state_3, done_3 = self.trunc_transition(state_3, action_3, reward_3, next_state_3, done_3)


                weight_state = torch.cat([state_2,state_3])
                weight_state = weight_state.reshape(-1, self.state_dim)
                weight_state = torch.cat([weight_state,task_state],axis=1).to(self.device)

                weight_next_state = torch.cat([next_state_2,next_state_3])
                weight_next_state = weight_next_state.reshape(-1, self.state_dim)
                weight_next_state = torch.cat([weight_next_state,task_state],axis=1).to(self.device)

                weight_action = torch.cat([action_2,action_3])
                weight_action = weight_action.reshape(-1,self.action_dim).to(self.device)

                # state = torch.cat([state_1,state_2,state_3])
                # state = state.reshape(-1, self.state_dim)
                state = torch.cat([state_1,task_state],axis=1).to(self.device)

                # next_state = torch.cat([next_state_1,next_state_2,next_state_3])
                # next_state = next_state.reshape(-1, self.state_dim)
                next_state = torch.cat([next_state_1,task_state],axis=1).to(self.device)

                # action = torch.cat([action_1,action_2,action_3])
                # action = action.reshape(-1,self.action_dim).to(self.device)

                reward = torch.cat([reward_1,reward_2,reward_3])
                # reward = reward.unsqueeze(1).to(self.device)
                done = torch.cat([done_1,done_2,done_3])
                # done = done.unsqueeze(1).to(self.device)
                action = action_1
                # reward = reward_1
                # done = done_1


            elif self.task_ind == 1 :
                reward_1 = backward_reward_1
                reward_2 = backward_reward_2
                reward_3 = backward_reward_3

                state_1, action_1, reward_1, next_state_1, done_1 = self.trunc_transition(state_1, action_1, reward_1, next_state_1, done_1)
                state_3, action_3, reward_3, next_state_3, done_3 = self.trunc_transition(state_3, action_3, reward_3, next_state_3, done_3)


                weight_state = torch.cat([state_1,state_3])
                weight_state = weight_state.reshape(-1, self.state_dim)
                weight_state = torch.cat([weight_state,task_state],axis=1).to(self.device)

                weight_next_state = torch.cat([next_state_1,next_state_3])
                weight_next_state = weight_next_state.reshape(-1, self.state_dim)
                weight_next_state = torch.cat([weight_next_state,task_state],axis=1).to(self.device)

                weight_action = torch.cat([action_1,action_3])
                weight_action = weight_action.reshape(-1,self.action_dim).to(self.device)

                # state = torch.cat([state_2,state_1,state_3])
                # state = state.reshape(-1, self.state_dim)
                state = torch.cat([state_2,task_state],axis=1).to(self.device)

                # next_state = torch.cat([next_state_2,next_state_1,next_state_3])
                # next_state = next_state.reshape(-1, self.state_dim)
                next_state = torch.cat([next_state_2,task_state],axis=1).to(self.device)
                action = action_2
                # reward = reward_2
                # done = done_2
                # action = torch.cat([action_2,action_1,action_3])
                # action = action.reshape(-1,self.action_dim).to(self.device)

                reward = torch.cat([reward_2,reward_1,reward_3])
                # reward = reward.unsqueeze(1).to(self.device)
                done = torch.cat([done_2,done_1,done_3])
                # done = done.unsqueeze(1).to(self.device)


            elif self.task_ind == 2 :
                reward_1 = jump_reward_1
                reward_2 = jump_reward_2
                reward_3 = jump_reward_3

                state_1, action_1, reward_1, next_state_1, done_1 = self.trunc_transition(state_1, action_1, reward_1, next_state_1, done_1)
                state_2, action_2, reward_2, next_state_2, done_2 = self.trunc_transition(state_2, action_2, reward_2, next_state_2, done_2)

                weight_state = torch.cat([state_1,state_2])
                weight_state = weight_state.reshape(-1, self.state_dim)
                weight_state = torch.cat([weight_state,task_state],axis=1).to(self.device)

                weight_next_state = torch.cat([next_state_1,next_state_2])
                weight_next_state = weight_next_state.reshape(-1, self.state_dim)
                weight_next_state = torch.cat([weight_next_state,task_state],axis=1).to(self.device)

                weight_action = torch.cat([action_1,action_2])
                weight_action = weight_action.reshape(-1,self.action_dim).to(self.device)

                # state = torch.cat([state_3,state_1,state_2])
                # state = state.reshape(-1, self.state_dim)
                state = torch.cat([state_3,task_state],axis=1).to(self.device)

                # next_state = torch.cat([next_state_3,next_state_1,next_state_2])
                # next_state = next_state.reshape(-1, self.state_dim)
                next_state = torch.cat([next_state_3,task_state],axis=1).to(self.device)
                action = action_3
                # reward = reward_3
                # done = done_3

                # action = torch.cat([action_3,action_1,action_2])
                # action = action.reshape(-1,self.action_dim).to(self.device)

                reward = torch.cat([reward_3,reward_1,reward_2])
                # reward = reward.unsqueeze(1).to(self.device)
                done = torch.cat([done_3,done_1,done_2])
                # done = done.unsqueeze(1).to(self.device)

            ## Weight
            if self.weight_method == 'cdsbasic' :
                relabel_weight = self.calcul_divergence_weight(task_state,weight_state,weight_action,weight_next_state)
                no_weight = torch.ones((int(self.batch_size/2)),1).to(self.device)
                relabel_weight_mean = relabel_weight.mean()
                relabel_weight = torch.cat([no_weight,relabel_weight])

            elif self.weight_method == 'cds' :
                relabel_weight = self.calcul_quantile_weight(task_state,weight_state,weight_action)
                no_weight = torch.ones((int(self.batch_size/2)),1).to(self.device)
                relabel_weight_mean = relabel_weight.mean()
                relabel_weight = torch.cat([no_weight,relabel_weight])

            else :
                raise e

            state = torch.cat([state,weight_state])
            state = state.reshape(-1,self.state_dim+self.task_num).to(self.device)
            next_state = torch.cat([next_state,weight_next_state])
            next_state = next_state.reshape(-1,self.state_dim+self.task_num).to(self.device)
            action = torch.cat([action,weight_action])
            action = action.reshape(-1,self.action_dim).to(self.device)


            # Actor Update
            current_alpha = copy.deepcopy(self.alpha)
            action_pred,log_prob_action = self.actor_network.evaluate(state)
            q1_val = self.critic_network_1(state, action_pred)
            q2_val = self.critic_network_2(state, action_pred)
            q_val = torch.min(q1_val, q2_val).cpu()
            actor_loss = relabel_weight.cpu() *(current_alpha * log_prob_action.cpu() - q_val)

            actor_loss = actor_loss.mean()

            # SAC Critic Update
            with torch.no_grad():
                next_action,next_log_prob_action = self.actor_network.evaluate(next_state)
                next_q_target_1 = self.critic_target_network_1(next_state, next_action)
                next_q_target_2 = self.critic_target_network_2(next_state, next_action)
                next_q_target = torch.min(next_q_target_1, next_q_target_2)
                next_q_target = next_q_target - self.alpha * next_log_prob_action
                q_target = self.reward_scale * reward + (self.discount_factor * next_q_target * (1 - done))

            q1_val = self.critic_network_1(state, action)
            q2_val = self.critic_network_2(state, action)

            critic_loss_1 = F.mse_loss(relabel_weight * q1_val, relabel_weight * q_target)
            critic_loss_2 = F.mse_loss(relabel_weight * q2_val, relabel_weight * q_target)


            # CQL 

            random_action = torch.FloatTensor(q1_val.shape[0] * 10, action.shape[-1]).uniform_(-1, 1).to(self.device)
            number_repeat = int(random_action.shape[0] / state.shape[0])
            temp_state = state.unsqueeze(1).repeat(1, number_repeat, 1).view(state.shape[0] * number_repeat, state.shape[1])
            temp_next_state = next_state.unsqueeze(1).repeat(1, number_repeat, 1).view(next_state.shape[0] * number_repeat,next_state.shape[1])

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

            cql_scaled_loss_1 = (((relabel_weight * (torch.logsumexp(cat_q_1 / self.temperature,
                                                  dim=1))).mean() * self.cql_weight * self.temperature) - (relabel_weight * q1_val).mean()) * self.cql_weight
            cql_scaled_loss_2 = (((relabel_weight * (torch.logsumexp(cat_q_2 / self.temperature,
                                                  dim=1))).mean() * self.cql_weight * self.temperature) - (relabel_weight * q2_val).mean()) * self.cql_weight


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


            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step() 

            self.critic_optimizer_1.zero_grad()
            total_critic_loss_1.backward(retain_graph=True)
            self.critic_optimizer_1.step()

            self.critic_optimizer_2.zero_grad()
            total_critic_loss_2.backward()
            self.critic_optimizer_2.step()

        self.load_dict()
        

        return min(total_critic_loss_1.item(),total_critic_loss_2.item()), actor_loss , relabel_weight_mean.item() , self.cql_log_alpha.item()
    

    def train_critic(self):
        ## TODO :
        raise NotImplementedError

    def train_actor(self):
        ## TODO :
        raise NotImplementedError

    def train_value(self):
        ## TODO :
        raise NotImplementedError