import numpy as np

MAX_VEL = 0.9

class RandVelGenerator:
    def __init__(self):
        self.jointDir = [False,False,False,False,False,False,False]

    def generatePatternVel(self):
        ranges = [0.6,0.6,0.7,0.8,0.8,0.8,0.8]
        output_vel = np.zeros(7)
        for i in range(7):
            output_vel[i] = ranges[i] * np.random.rand()
            if self.jointDir[i] == False:
                output_vel[i] = -output_vel[i]
        return output_vel

    def setJointDirections(self, directions):
        for i in range(7):
            if directions[i] > 0:
                self.jointDir[i] = False
            elif directions[i] < 0:
                self.jointDir[i] = True

    # def goingRightDirection(self, q):
