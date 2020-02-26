import numpy as np
from numpy import cos
from numpy import sin
import math


def get_aa(S,E,W):
    normal_vec = np.cross(E-S, E-W)
    V = np.array([0,0,1])
    return math.acos(np.dot(normal_vec, V)/(np.linalg.norm(normal_vec) * np.linalg.norm(V)))


def get_rotation_matrix(phi,theta,psi):
    return np.array([
    [cos(theta)*cos(phi),    sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi),   cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],
    [cos(theta)*sin(phi),    sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi),   cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi)],
    [-sin(theta),            sin(psi)*cos(theta),                                cos(psi)*cos(theta)]
    ])


def get_orientation(R):
    R11 = R[0,0]
    R12 = R[0,1]
    R13 = R[0,2]
    R21 = R[1,0]
    R22 = R[1,1]
    R23 = R[1,2]
    R31 = R[2,0]
    R32 = R[2,1]
    R33 = R[2,2]
    if R31 != 1 and R31 != -1:
        theta = -math.asin(R31)
        psi = math.atan2(R32/cos(theta), R33/cos(theta))
        phi = math.atan2(R21/np.cos(theta), R11/cos(theta))
    else:
        phi = 0
        if R31 == -1:
            theta = np.pi/2
            psi = phi + math.atan2(R12,R13)
        else:
            theta = -np.pi/2
            psi = -theta + math.atan2(-R12,-R13)
    return np.array([psi, theta, phi])


def get_3D_data(sim):
    S = sim.data.body_xpos[sim.model.body_name2id("link1")]
    E = sim.data.body_xpos[sim.model.body_name2id("link4")]
    W = sim.data.body_xpos[sim.model.body_name2id("link7")]
    EE = sim.data.body_xpos[sim.model.body_name2id("right_hand")]
    R = sim.data.body_xmat[sim.model.body_name2id("right_hand")].reshape([3,3])
    aa = get_aa(S,E,W)
    orientation = get_orientation(R)
    return np.concatenate((EE,orientation,[aa]))


def main():
    R = np.array([
    [0.5, -0.1464, 0.8536],
    [0.5, 0.8536, -0.1464],
    [-0.7071, 0.5, 0.5]
    ])
    print(get_orientation(R)/np.pi)
    print(get_rotation_matrix(np.pi/4, np.pi/4, np.pi/4))


if __name__ == '__main__':
    main()
