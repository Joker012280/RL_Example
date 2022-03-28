import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
class actor(nn.Module) :
    def __init__(self,input_size,hidden,output_size):
        super(actor,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,output_size)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))*2
        return x


class critic(nn.Module) :
    def __init__(self,input_size,hidden):
        super(critic,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        self.fc3 = nn.Linear(hidden,1)

    def forward(self,x,y):
        x = F.relu(self.fc1(torch.cat((x,y),dim =1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x