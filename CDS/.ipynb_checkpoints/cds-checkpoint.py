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
    def __init__(self, 
                 state_dim, 
                 hidden, 
                 action_dim,
                 weight_temperature,
                 action_bounds,
                 weight=None,
                 reward_scale = 1, 
                 tau=None, 
                 target_entropy=None, 
                 temperature=None, 
                 batch_size = 256,
                 task_num = 1,
                 task_idx = None,
                 cds_module = True,
                 with_lagrange = False,
                 data_sharing = True,
                 device = None,
                 ):
        super(CDS, self).__init__()
        

        self.task_idx = task_idx
        self.task_num = task_num
        # Condition Check
        if task_idx is None and task_num == 1 :
            raise e        
        # For single Task
        if self.task_idx is not None :
            task_num = 0   
            self.task_num = 1
        self.data_sharing = data_sharing
        self.actor_network = models.actor(state_dim + task_num, hidden, action_dim,action_bounds).to(device)
        self.critic_network_1 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_1 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_1.load_state_dict(self.critic_network_1.state_dict())
        self.critic_network_2 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        assert self.critic_network_1.parameters() != self.critic_network_2.parameters()
        self.critic_target_network_2 = models.critic(state_dim + action_dim + task_num, hidden).to(device)
        self.critic_target_network_2.load_state_dict(self.critic_network_2.state_dict())
        if self.task_num == 1 :
            self.actor_lr = 3e-4
        else :
            self.actor_lr = 1e-4
        self.critic_lr = 3e-4
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), self.actor_lr)
        self.critic_optimizer_1 = optim.Adam(self.critic_network_1.parameters(), self.critic_lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_network_2.parameters(), self.critic_lr)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        
 
        self.device = device
        
        # SAC & CQL
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = 1.0
        # self.alpha_lr = 3e-4
        # self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.alpha_lr)
        self.cql_weight = 1.0
        self.temperature = 5.0 if temperature is None else temperature
        self.reward_scale = reward_scale
        # Target Action Gap Change
        self.target_action_gap = 30.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=self.critic_lr) 
        self.with_lagrange = with_lagrange
        self.tau = 0.005 if tau is None else tau
        
        # CDS For MultiTask 
        self.weight_method = weight
        self.discount_factor = 0.99
        self.weight_temperature = weight_temperature
        self.current_weight_temperature = None
        self.batch_size = batch_size
        
        
        self.memory = Buffer.Replay_buffer(self.batch_size)
        
    
    
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
        
    def calcul_divergence_weight(self,ori_state,ori_action,ori_next_state,state,action,next_state) :
        
        ori_conservative_qval = self.get_conservative_qval(ori_state,ori_action,ori_next_state)
        diff_task_conservative_qval = self.get_conservative_qval(state,action,next_state)
        
        delta_term = ori_conservative_qval - diff_task_conservative_qval
        # delta_term = diff_task_conservative_qval - ori_conservative_qval
        
        
        if self.current_weight_temperature is None :
            self.current_weight_temperature = delta_term.mean()
        
        weight = torch.sigmoid(delta_term / self.current_weight_temperature)
        self.cds_weight_decay(delta_term.mean())
        
        return weight.to(self.device)

    def calcul_quantile_weight(self,ori_state,ori_action,state,action):
        
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
        
        return weight.to(self.device)
    
    def cat_sample(self,*samples) :
        
        state, action, jump_reward,forward_reward,backward_reward, next_state, done = zip(*samples)
        state = torch.cat(state)
        state = state.reshape(-1, self.state_dim)
        next_state = torch.cat(next_state)
        next_state = next_state.reshape(-1, self.state_dim)
        action = torch.cat(action)
        action = action.reshape(-1,self.action_dim)
        jump_reward = torch.cat(jump_reward)
        jump_reward = jump_reward.unsqueeze(1)
        forward_reward = torch.cat(forward_reward)
        forward_reward = forward_reward.unsqueeze(1)
        backward_reward = torch.cat(backward_reward)
        backward_reward = backward_reward.unsqueeze(1)
        done = torch.cat(done)
        done = done.unsqueeze(1)
        
        return state, action, jump_reward,forward_reward,backward_reward, next_state, done

    
    def trunc_transition(self,state,action,reward,next_state,done,divide_num) :
        
        divide_num = int(divide_num)
        state = torch.chunk(state,divide_num,dim=0)
        action = torch.chunk(action,divide_num,dim=0)
        reward = torch.chunk(reward,divide_num,dim=0)
        next_state = torch.chunk(next_state,divide_num,dim=0)
        done = torch.chunk(done,divide_num,dim=0)
        
        return state[0],action[0],reward[0],next_state[0],done[0]

    def get_weight_and_samples(self,rand_data_idx):
        
        ## One hot state for Multi task
        if rand_data_idx == 0 and self.task_num != 1 :
            task_state = np.array([[1,0,0]])
            task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0))
        elif rand_data_idx == 1 and self.task_num != 1:
            task_state = np.array([[0,1,0]])
            task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0))
        elif rand_data_idx == 2 and self.task_num != 1:
            task_state = np.array([[0,0,1]])
            task_state = torch.from_numpy(np.repeat(task_state,int(self.batch_size/2),axis=0))
        else :
            task_state = torch.FloatTensor([])
       
        weight_state = torch.FloatTensor([])
        weight_next_state = torch.FloatTensor([])
        weight_action = torch.FloatTensor([])
        weight_reward = torch.FloatTensor([])
        weight_done = torch.FloatTensor([])
        
        
        ## Sampling Data from Each task
        for task in range(self.task_num) :
            samples = self.memory.sample(task)
            # Multiple Reward (TODO Change it more simpler)
            s,a,j_r,f_r,b_r,ns,d = self.cat_sample(*samples)
            
            # Reward Selecting
            if (rand_data_idx == 0 and self.task_num != 1) or (self.task_idx == 0) :
                r = f_r
            elif (rand_data_idx == 1 and self.task_num != 1) or (self.task_idx == 1) :
                r = b_r
            elif (rand_data_idx ==2 and self.task_num != 1) or (self.task_idx == 2) :
                r = j_r
            else :
                raise e
                
            if task == rand_data_idx : # For specific task
                state,action,reward,next_state,done = self.trunc_transition(s,a,r,ns,d,2)
            
            else : # For different task
                s,a,r,ns,d = self.trunc_transition(s,a,r,ns,d,4)
                weight_state = torch.cat([weight_state,s])
                weight_next_state = torch.cat([weight_next_state,ns])
                weight_action = torch.cat([weight_action,a])
                weight_reward = torch.cat([weight_reward,r])
                weight_done = torch.cat([weight_done,d])
                
        ## TO GPU
        state = torch.cat([state,task_state],axis=1).to(self.device)
        next_state = torch.cat([next_state,task_state],axis=1).to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        relabel_weight = torch.FloatTensor([1]).to(self.device)
        relabel_weight_mean = torch.FloatTensor([1]).to(self.device)
        
        if self.data_sharing :
            weight_state = torch.cat([weight_state,task_state],axis=1).to(self.device)
            weight_next_state = torch.cat([weight_next_state,task_state],axis=1).to(self.device)
            weight_action = weight_action.to(self.device)
            weight_reward = weight_reward.to(self.device)
            weight_done = weight_done.to(self.device)
        
            ## Weight
            if self.weight_method == 'cdsbasic' :
                relabel_weight = self.calcul_divergence_weight(state,action,next_state,weight_state,weight_action,weight_next_state)
                no_weight = torch.ones((int(self.batch_size/2)),1).to(self.device)
                relabel_weight_mean = relabel_weight.mean()
                relabel_weight = torch.cat([no_weight,relabel_weight])

            elif self.weight_method == 'cds' :
                relabel_weight = self.calcul_quantile_weight(state,action,weight_state,weight_action)
                no_weight = torch.ones((int(self.batch_size/2)),1).to(self.device)
                relabel_weight_mean = relabel_weight.mean()
                relabel_weight = torch.cat([no_weight,relabel_weight])

            else : # no weight
                relabel_weight = torch.ones((int(self.batch_size)),1).to(self.device)
                relabel_weigth_mean = relabel_weight.mean()

            ## Transition Concat

            state = torch.cat([state,weight_state]).to(self.device)
            action = torch.cat([action,weight_action]).to(self.device)
            reward = torch.cat([reward,weight_reward]).to(self.device)
            next_state = torch.cat([next_state,weight_next_state]).to(self.device)
            done = torch.cat([done,weight_done]).to(self.device)

        return state,action,reward,next_state,done,relabel_weight,relabel_weight_mean
    
        
    
    def train_net(self):
        # Random Sample Task 
        # For all task make update 
        # Task 0 : Forward, 1 : Backward, 2 : Jump
        
        rand_data_idx = random.randrange(len(self.memory.memory.keys()))
                 
            
        state,action,reward,next_state,done,relabel_weight,relabel_weight_mean = self.get_weight_and_samples(rand_data_idx)

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