"""
Data structure for implementing experience replay
Author: Kiagus Arief Adriansyah
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward, done, next_state):
        experience = (state, action, reward, done, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        state_batch      = np.array([_[0] for _ in batch])
        action_batch     = np.array([_[1] for _ in batch])
        reward_batch     = np.array([_[2] for _ in batch])
        done_batch       = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
