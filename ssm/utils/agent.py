import grid2op
from grid2op.Agent import BaseAgent
import random

class RandomTopologyAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.action_space = action_space

    def act(self):
        return random.choice(self.action_space)