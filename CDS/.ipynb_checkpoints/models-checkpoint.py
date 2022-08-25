import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)
        
class actor(nn.Module):
    def __init__(self, input_size, hidden, output_size, action_bounds,log_std_min=-20, log_std_max=2):
        super(actor, self).__init__()
        
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(input_size, hidden)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        
        self.fc2 = nn.Linear(hidden, hidden)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        
        self.fc3 = nn.Linear(hidden, output_size)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()
        
        self.fc4 = nn.Linear(hidden, output_size)
        init_weight(self.fc4, initializer="xavier uniform")
        self.fc4.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.fc4(x)
        std = torch.exp(torch.clamp(log_std, self.log_std_min, self.log_std_max))

        return mean, std
    
    def get_action(self,x):
        mean,std = self.forward(x)
        action_dist = Normal(mean,std)
        action = action_dist.rsample()
        return action.detach()
    
    def evaluate(self, x):
        mean, std = self.forward(x)
        action_dist = Normal(mean, std)
        e = action_dist.rsample()
        action = torch.tanh(e)
        log_prob = (action_dist.log_prob(e)-torch.log(1 - action.pow(2) + 1e-6)).sum(1,keepdim=True)
        if self.action_bounds is not None :
            action = torch.clamp(action,min=self.action_bounds[0],max=self.action_bounds[1])
        return action, log_prob

class behavior(nn.Module):
    def __init__(self, input_size, hidden, output_size, action_bounds,log_std_min=-20, log_std_max=2):
        super(behavior, self).__init__()
        
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(input_size, hidden)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        
        self.fc2 = nn.Linear(hidden, hidden)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        
        self.fc3 = nn.Linear(hidden, output_size)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()
        
        self.fc4 = nn.Linear(hidden, output_size)
        init_weight(self.fc4, initializer="xavier uniform")
        self.fc4.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.fc4(x)
        std = torch.exp(torch.clamp(log_std, self.log_std_min, self.log_std_max))

        return mean, std
    
    def get_action(self,x):
        mean,std = self.forward(x)
        action_dist = Normal(mean,std)
        action = action_dist.rsample()
        return action.detach()
    
    def evaluate(self, x):
        mean, std = self.forward(x)
        action_dist = Normal(mean, std)
        e = action_dist.rsample()
        action = torch.tanh(e)
        if self.action_bounds is not None :
            action = torch.clamp(action,min=self.action_bounds[0],max=self.action_bounds[1])
        return action, action_dist
    
class actor_dist(nn.Module):
    def __init__(self, input_size, hidden, output_size, action_bounds,log_std_min=-20, log_std_max=2):
        super(actor_dist, self).__init__()
        
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(input_size, hidden)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        
        self.fc2 = nn.Linear(hidden, hidden)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        
        self.fc3 = nn.Linear(hidden, output_size)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()
        
        self.fc4 = nn.Linear(hidden, output_size)
        init_weight(self.fc4, initializer="xavier uniform")
        self.fc4.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.fc4(x)
        std = torch.exp(torch.clamp(log_std, self.log_std_min, self.log_std_max))

        return mean, std
    
    def get_action(self,x):
        mean,std = self.forward(x)
        action_dist = Normal(mean,std)
        action = action_dist.rsample()
        return action.detach()
    
    def evaluate(self, x):
        mean, std = self.forward(x)
        action_dist = Normal(mean, std)
        e = action_dist.rsample()
        action = torch.tanh(e)
        log_prob = (action_dist.log_prob(e)-torch.log(1 - action.pow(2) + 1e-6)).sum(1,keepdim=True)
        if self.action_bounds is not None :
            action = torch.clamp(action,min=self.action_bounds[0],max=self.action_bounds[1])
        return action, log_prob , action_dist
    
    
class actor_discrete(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(actor_discrete, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def evaluate(self, x):
        logit = self.forward(x)
        dist = Categorical(logits=logit)
        action = dist.sample()
        log_prob = (dist.log_prob(action)-torch.log(1-action.pow(2) + 1e-6)).sum(1,keepdim=True)
        
        return action, log_prob


class critic(nn.Module):
    def __init__(self, input_size, hidden):
        super(critic, self).__init__()
        # torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        
        self.fc2 = nn.Linear(hidden, hidden)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        
        self.fc3 = nn.Linear(hidden, 1)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()
        
    def forward(self, x, y):
        x = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class critic_discrete(nn.Module):
    def __init__(self, input_size, hidden,action_dim):
        super(critic_discrete, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class value(nn.Module):
    def __init__(self, input_size, hidden):
        super(value, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        init_weight(self.fc1)
        self.fc1.bias.data.zero_()
        
        self.fc2 = nn.Linear(hidden, hidden)
        init_weight(self.fc2)
        self.fc2.bias.data.zero_()
        
        self.fc3 = nn.Linear(hidden, 1)
        init_weight(self.fc3, initializer="xavier uniform")
        self.fc3.bias.data.zero_()
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x