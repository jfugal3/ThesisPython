import numpy as np


UPPER_BOUND = [2.6973, 1.5628, 2.6973, -0.2698, 2.6973, 3.5525, 2.6973]
LOWER_BOUND = [-2.6973, -1.5628, -2.6973, -2.700, -2.6973, 1.5, -2.6973]

def getBoundViolations(q):
    jointViolations = np.array([0,0,0,0,0,0,0])
    for i in range(7):
        if q[i] < LOWER_BOUND[i]:
            jointViolations[i] = -1
        elif q[i] > UPPER_BOUND[i]:
            jointViolations[i] = 1
    return jointViolations

def tableBoundViolation(sim):
    THRESHOLD = 0.26
    frame_names = ["link1","link2","link3","link4","link5","link6","link7","right_hand"]
    for name in frame_names:
        zi = sim.data.body_xpos[sim.model.body_name2id(name)][2]
        if zi < 0.26:
            return True

    return False


def outOfBounds(q):
    for i in range(7):
        if UPPER_BOUND[i] < q[i] or q[i] < LOWER_BOUND[i]:
            return True

    return False

def getRandPosInBounds():
    ub = np.array(UPPER_BOUND)
    lb = np.array(LOWER_BOUND)
    jointRanges = ub - lb
    return lb + np.random.rand(7) * jointRanges
