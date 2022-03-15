import torch
import random

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

    def size(self):
        return self.buffer_size