import numpy as np
import datetime
import math


def moving(qd):
    return np.linalg.norm(qd) > 0.1

def stopped(qd):
    return np.linalg.norm(qd) < 0.01

def getUniqueFileName(const_str):
    return const_str + "_" + str(datetime.datetime.now())


class StringException(Exception):
    def __init__(self, message):
        self.message = message

    def what(self):
        return self.message

grav_options = {
    "perfect_comp" : 0,
    "no_comp" : 1,
    "ee_PD_cont" : 2,
    "q_PD_cont" : 3
}


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def normalize_sym(x, low, high):
    # place in range from [-1, 1]
    if np.any(high <= low):
        print("helperFun.normalize_sym: range exception!")
        print("low =", low)
        print("high =", high)
    return 2 * (x - low) / (high - low) - 1

def unnormalize_sym(x, low, high):
    if np.any(high <= low):
        print("helperFun.unnormalize_sym: range exception!")
        print("low =", low)
        print("high =", high)
    return (x + 1)*(high - low)/2 + low

def normalize(x, low, high):
    return (x - low) / (high - low)

def unnormalize(x, low, high):
    return x * (high - low) + low
