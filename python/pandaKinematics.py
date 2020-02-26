import numpy as np
from numpy import pi
from numpy import cos
from numpy import sin
from math import atan2

class kinematics:
    def __init__(self):
        self.a = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
        self.d = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
        self.alpha = np.array([0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0])

    def T(self, joint, theta):
        a = self.a[joint]
        d = self.d[joint]
        alpha =  self.alpha[joint]
        return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha),  cos(alpha), d],
        [0, 0, 0, 1]
        ])

    def get3Dpose(self, joint, q):
        q_aug = np.append(q,0)
        H = np.identity(4)
        for i in range(joint):
            H = H @ self.T(i, q_aug[i])
        return H

def main():
    q = np.array([0.0121, -0.7754, -0.0220, -3.0432, -0.1600, 2.0169, 0.7959])
    k = kinematics()
    for i in range(1,9):
        print("count:",i-1)
        print(k.get3Dpose(i, q))

if __name__ == '__main__':
    main()
