import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
class actor(nn.Module) :
    def __init__(self,input_size,hidden,output_size,log_std_min = -10,log_std_max = 2):
        super(actor,self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,output_size)
        self.fc4 = nn.Linear(hidden,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        log_std = self.fc4(x)
        std = torch.exp(torch.clamp(log_std,self.log_std_min,self.log_std_max))
        dist = Normal(mean,std)
        return dist


class critic(nn.Module) :
    def __init__(self,input_size,hidden,seed =1):
        super(critic,self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,1)

    def forward(self,x,y):
        x = F.relu(self.fc1(torch.cat((x,y),dim =1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class value(nn.Module) :
    def __init__(self,input_size,hidden):
        super(value,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x