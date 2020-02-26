from ikpy.chain import Chain
from ikpy.link import URDFLink
from ikpy import geometry_utils
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
# my_chain = Chain.from_urdf_file("robot_urdf.xml")
# name (str) – The name of the link
# bounds (tuple) – Optional : The bounds of the link. Defaults to None
# translation_vector (numpy.array) – The translation vector. (In URDF, attribute “xyz” of the “origin” element)
# orientation (numpy.array) – The orientation of the link. (In URDF, attribute “rpy” of the “origin” element)
# rotation (numpy.array) – The rotation axis of the link. (In URDF, attribute “xyz” of the “axis” element)
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

my_chain = Chain([link0, link1, link2, link3, link4, link5, link6, link7, link8])


roll = 0 # rotation about z
pitch = 0 # rotation about y
yaw = 0 # rotation about x

x = 0.5
y = -0.15
z = 1.4

v = np.array([x,y,z])
R = geometry_utils.rpy_matrix(roll=roll, pitch=pitch, yaw=yaw)
T = geometry_utils.to_transformation_matrix(translation=v, orientation_matrix=R)

theta = my_chain.inverse_kinematics(target=T, initial_position=np.zeros(9))
print("joint angles:",theta)

# print(my_chain.forward_kinematics([0]*9))

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
my_chain.plot(np.zeros(9), ax, show=True)
# my_chain.plot(0,)
