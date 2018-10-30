import numpy as np
from collections import deque


class ReplayBuffer(object):

    def __init__(self, size):
        self.memory_max_size = size
        self.memory = deque(maxlen=self.memory_max_size)
        self.counter = 0

    def remember(self, transitions):
        if len(self.memory) >= self.memory_max_size:
            self.memory.popleft()
        self.memory.append(transitions)
        self.counter = self.counter+1

    def sample_from_memory(self):
        index = np.random.randint(0, len(self.memory), 1)
        return self.memory[index[0]]

    def can_sample(self):
        if len(self.memory) < self.memory_max_size/3:
            return False
        return True
