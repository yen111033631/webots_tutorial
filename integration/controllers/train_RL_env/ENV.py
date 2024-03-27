from ikpy.chain import Chain
from controller import Supervisor
import numpy as np
import time
import random



class env_ball:
    def __init__(self):
        self.obvervation = [[], []]
        self.info = "ball"
        self.ball = Object("ball")



    def reset(self, seed=0):
        random.seed(seed)
        self.ball.random_choose_one_position()
        self.ball.setP(self.ball.new_position)
        print(self.ball.new_position)

        self.

        self.obvervation[1] = self.ball.position



        return self.obvervation, self.info




    def step(self, action):
        pass



class Object:
    def __init__(self, DEF_name):
        self.name = DEF_name
        self.node = Supervisor().getFromDef(DEF_name)
        self.translation_field = self.node.getField('translation')
        self.position = self.node.getPosition()
        self.position_candididate = [[-0.23, 0.5, 0.24],
                                     [0.42, 0.5, 0.24],
                                     [0.02, 0.45, 0.86],
                                     [0.02, 0.35, 0.6],
                                     [0.02, -0.44095, 0.16704],
                                     [0.195, -0.22661, 0.38138],
                                     [0.37871, -0.62161, 0.62638],
                                     [0.20724, -0.57161, 0.71638]]
        self.new_position = None

    def read_position(self):
        self.position = self.node.getPosition()
    
    def random_choose_one_position(self):
            random_index = random.randint(0, len(self.position_candididate) - 1)
            self.new_position = self.position_candididate[random_index]

    def setP(self, position):
            self.translation_field.setSFVec3f(position)
            self.position = position
            # self.read_position()

if __name__ == "__main__":

    observation, info = env.reset(seed=seed)

    next_observation, reward, terminated, truncated, _ = env.step(action)