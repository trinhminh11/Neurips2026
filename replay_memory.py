import random
import numpy as np

class Transition:
    def __init__(self, state: np.ndarray, z: int, done: bool, action: int, next_state: np.ndarray, prev = None):
        self.state = state
        self.z = z
        self.done = done
        self.action = action
        self.next_state = next_state
        self.prev = prev

    def traverse(self):
        res = []
        node = self
        while node is not None:
            res.append(node)
            node = node.prev
        return res[::-1]    # reverse to make the order from the earliest transition to the latest transition


class Memory:
    def __init__(self, buffer_size: int, seed: int):
        self.buffer_size = buffer_size
        self.buffer: list[Transition] = []
        self.seed = seed
        random.seed(self.seed)

    def add(self, state: np.ndarray, z: int, done: bool, action: int, next_state: np.ndarray, skill_start: bool = False):
        if len(self.buffer) == 0 or skill_start is True:
            prev = None
        else:
            prev = self.buffer[-1]

        self.buffer.append(Transition(state, z, done, action, next_state, prev=prev))

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    def sample(self, size: int):
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def get_rng_state() -> tuple:
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state: tuple):
        random.setstate(random_rng_state)
