import random
from collections import deque

class Replay:
    def __init__(self, buffer_size):
        self.buffer = deque(max_len = buffer_size)

    # state, action, reward, next_state, done
    def add(self, event):
        self.buffer.append(event)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    