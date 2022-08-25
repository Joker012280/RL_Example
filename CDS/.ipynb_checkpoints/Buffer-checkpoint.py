import torch
import random
import pickle

class Replay_buffer(object):
    def __init__(self,batch_size,task_num = 0,capacity = None):
        
        self.task_num = task_num
        if task_num != 0 :
            self.memory = {t:[] for t in range(task_num)}
        else :
            self.memory = {0:[]}
        self.batch_size = batch_size
        self.capacity = capacity

    def push(self, data,task_num=0):
        if self.capacity is not None :
            self.memory[task_num].append(data)
            if len(self.memory[task_num]) > self.capacity:
                self.memory[task_num].pop(0)
            assert len(self.memory) <= self.capacity
        else :
            self.memory[task_num].append(data)


    def sample(self,task_num=0):
        return random.sample(self.memory[task_num],self.batch_size)

    def clear(self):
        self.memory = []

    def save_data(self,name,task_num = 0):
        with open(str(name)+".pickle","wb") as fw:
            pickle.dump(self.memory[task_num],fw)
    def load_data(self,name,task_num = 0):
        with open(str(name)+".pickle",'rb') as f:
            self.memory[task_num] = pickle.load(f)

    def size(self,task_num=0):
        return len(self.memory[task_num])