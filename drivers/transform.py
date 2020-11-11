import math
import numpy as np
from envs import minitaur

NUM_LEGS=4

class Transform():
    def __init__(self):
        self.trot_step_frequency=2
        self._extension_amplitude=0.35
        self._swing_amplitude=0.3
        self._swing_offset=0
        self._extension_offset=0


