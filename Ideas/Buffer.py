import random
import torch

class Replay_buffer():
    def __init__(self, max_size=1000):
        self.memory = []
        self.max_size = max_size
        self.position = 0
        self.buffer_size = 0

    def push(self, data):
        if len(self.memory) == self.max_size:
            self.memory[int(self.position)] = data
            self.position = (self.position + 1) % self.max_size
        else:
            self.memory.append(data)
            self.buffer_size += 1

    def sample(self):
        old_actions,probs,states= torch.FloatTensor(),torch.FloatTensor(), torch.FloatTensor()
        next_states, rewards, done = [], [], []

        ## 받은 샘플들을 합쳐서 내보냄
        for i in range(self.buffer_size):
            old_action,prob,state,next_state, reward, done_ = self.memory[i]
            old_actions = torch.cat((old_actions,old_action))
            probs = torch.cat((probs,prob))
            states = torch.cat((states,state))
            next_states.append([next_state])
            rewards.append([reward])
            done.append([done_])

        next_states = torch.FloatTensor(next_states)
        ## Return 값이 각 요소에 따른 텐서들을 전달
        return old_actions.detach(),probs,states.detach(),  \
               next_states, torch.FloatTensor(rewards), torch.FloatTensor(done)

    def clear(self):
        self.memory = []
        self.position = 0
        self.buffer_size = 0

    def size(self):
        return self.buffer_size