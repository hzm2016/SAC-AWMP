import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def add_final_reward(self, final_reward, steps, delay=0):
        len_buffer = len(self.storage)
        for i in range(len_buffer - steps - delay, len_buffer - delay):
            item = list(self.storage[i])
            item[3] += final_reward
            self.storage[i] = tuple(item)

    def add_specific_reward(self, reward_vec, idx_vec):
        for i in range(len(idx_vec)):
            time_step_num = int(idx_vec[i])
            item = list(self.storage[time_step_num])
            item[3] += reward_vec[i]
            self.storage[time_step_num] = tuple(item)

    def __len__(self):
        return len(self.storage)
