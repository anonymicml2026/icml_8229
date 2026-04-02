import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action,reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) 
        self.buffer[self.position] = (state, action, reward,next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def out(self):
        trajectory_list=[]
        trajectory_individual_list = []
        print("replay_buffer_length:",len(self.buffer))
        
        for i in range(len(self.buffer)-1,0,-1):
            trajectory_individual_list.append(self.buffer[i])
            print("self.buffer[i][4]:",self.buffer[i][4])
            if self.buffer[i][4] == 0:
               trajectory_list.append(torch.Tensor(trajectory_individual_list))
               trajectory_individual_list = []
        return torch.stack(trajectory_list)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action,reward, next_state, done= map(np.stack, zip(*batch))
        return state, action,reward, next_state, done

    def __len__(self):
        return len(self.buffer)
