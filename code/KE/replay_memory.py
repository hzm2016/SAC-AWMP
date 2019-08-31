import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[0:-1] = self.buffer[1:]
            self.buffer[-1] = (state, action, reward, next_state, done)

    def add_final_reward(self, final_reward, steps):
        len_buffer = len(self.buffer)
        for i in range(len_buffer - steps, len_buffer):
            item = list(self.buffer[i])
            item[2] += final_reward
            self.buffer[i] = tuple(item)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
