import random
from collections import deque

class Replay:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size

    # state, action, reward, next_state, done
    def add(self, event):
        self.buffer.append(event)

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    