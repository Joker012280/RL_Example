import torch
import random

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        self.position = (self.position + 1) % self.capacity
        if len(self.memory) >= self.capacity:
            self.memory[self.position] = transition
            print("Maximum Memory Capacity")

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
