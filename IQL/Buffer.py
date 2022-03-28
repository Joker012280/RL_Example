import torch
import random
import pickle

class Replay_buffer(object):
    def __init__(self,batch_size,capacity = None):
        self.memory = []
        self.position = 0
        self.buffer_size = 0
        self.batch_size = batch_size
        self.capacity = capacity



    def push(self, data):
        if self.capacity is not None :
            if len(self.memory) == self.max_size:
                self.memory[int(self.position)] = data
                self.position = (self.position + 1) % self.max_size
            else:
                self.memory.append(data)
                self.buffer_size += 1
        else :
            self.memory.append(data)
            self.buffer_size += 1

    def sample(self):
        return random.sample(self.memory,self.batch_size)

    def clear(self):
        self.memory = []
        self.position = 0
        self.buffer_size = 0

    def save_data(self):
        with open("data2.pickle","wb") as fw:
            pickle.dump(self.memory,fw)
    def load_data(self):
        with open("data2.pickle",'rb') as f:
            self.memory = pickle.load(f)

    def size(self):
        return len(self.memory)