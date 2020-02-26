from ikpy.chain import Chain
from ikpy.link import URDFLink
from ikpy import geometry_utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import data_calc
import math


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

class panda_kinematics:
    def __init__(self):
        link0 = URDFLink(name="link0", bounds=(0,0),
                        translation_vector=np.zeros(3),
                        orientation=np.zeros(3),
                        rotation=np.zeros(3))

        link1 = URDFLink(name="link1", bounds=(-2.8973,2.8973),
                        translation_vector=np.array([0, 0, 0.333]),
                        orientation=np.array([0, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link2 = URDFLink(name="link2", bounds=(-1.7628,1.7628),
                        translation_vector=np.array([0, 0, 0]),
                        orientation=np.array([-np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link3 = URDFLink(name="link3", bounds=(-2.8973,2.8973),
                        translation_vector=np.array([0, -0.316, 0]),
                        orientation=np.array([np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link4 = URDFLink(name="link4", bounds=(-3.0718,-0.0698),
                        translation_vector=np.array([0.0825, 0, 0]),
                        orientation=np.array([np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link5 = URDFLink(name="link5", bounds=(-2.8973, 2.8973),
                        translation_vector=np.array([-0.0825, 0.384, 0]),
                        orientation=np.array([-np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link6 = URDFLink(name="link6", bounds=(-0.0175, 3.7525),
                        translation_vector=np.array([0, 0, 0]),
                        orientation=np.array([np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link7 = URDFLink(name="link7", bounds=(-2.8973, 2.8973),
                        translation_vector=np.array([0.088, 0, 0]),
                        orientation=np.array([np.pi/2, 0, 0]),
                        rotation=np.array([0, 0, 1]))

        link8 = URDFLink(name="link8", bounds=(0,0),
                        translation_vector=np.array([0, 0, 0.107]),
                        orientation=np.array([0, 0, 0]),
                        rotation=np.array([0, 0, 0]))

        self.my_chain = Chain([link0, link1, link2, link3, link4, link5, link6, link7, link8])

    def inverse_kinematics(self, translation, rpy, init_qpos=np.zeros(7)):
        R = geometry_utils.rpy_matrix(roll=rpy[0], pitch=rpy[1], yaw=rpy[2])
        # print(R)
        T = geometry_utils.to_transformation_matrix(translation=translation, orientation_matrix=R)
        initial_position = np.concatenate(([0],init_qpos,[0]))
        return self.my_chain.inverse_kinematics(target=T, initial_position=initial_position)[1:8]

    def forward_kinematics(self, qpos):
        q = np.concatenate(([0], qpos, [0]))
        v, R = geometry_utils.from_transformation_matrix(self.my_chain.forward_kinematics(q))
        v = v[:-1]
        orientation = rotationMatrixToEulerAngles(R)
        roll, pitch, yaw = orientation[2], orientation[1], orientation[0]
        return np.concatenate((v, [roll, pitch, yaw]))

    def plot_stick_figure(self, qpos):
        q = np.concatenate(([0], qpos, [0]))
        fig = plt.figure(2020)
        ax = fig.add_subplot(1,1,1, projection='3d')
        self.my_chain.plot(q, ax, show=True)

    def euler_angles_to_rpy_rotation_matrix(self, rpy):
        return geometry_utils.rpy_matrix(roll=rpy[0], pitch=rpy[1], yaw=rpy[2])
