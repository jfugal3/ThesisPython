import numpy as np
import helperFun


  # d<< 20, 15, 15, 10, 10, 10.0, 5.0;
  # Matrix7d D = 0.05 * d.asDiagonal();
  # double alpha = 10;
  # Matrix7d Kp = alpha*D;
  # Array77d D_array(D.data());
  # Matrix7d sqrtD(D_array.sqrt());
  # Matrix7d Kd = 2*sqrt(alpha)*sqrtD;
  # Vector7d tau = Kp*(qgoal_eig - q_eig) - Kd*qd_eig;


def PDControl(q, qd, qgoal):
    Kp = 10 * np.diag([15, 13, 11, 10, 8, 4, 1])
    Kd = 3 * np.diag([15, 13, 11, 10, 8, 4, 1])
    tau = Kp @ (qgoal - q) - Kd @ qd
    return tau



def basicVelControl(qd_des, qd_cur):
    K = 2 * np.array([13,8,6,2,1.5,1.5,1])
    return np.diag(K) @ (qd_des - qd_cur)


def dampingControl(qd):
    K = np.array([30.0, 30.0, 30.0, 30.0, 5.0, 1.0, 1.0])
    return -np.diag(K) @ qd


def moveAwayFromTable(q,qd):
    qgoal = q;
    qgoal[1] = -0.0125
    qgoal[3] = -0.1780
    return PDControl(q,qd,qgoal)


# def moveToPos(sim, qgoal):
#     done = False
#     time = 0
#     while not done:
#         state = sim.get_state()
#         q = state[1]
#         qd = state[2]
#         sim.data.ctrl[:] = PDControl(q,qd,qgoal)
#         sim.step()
#         time +=
#         done = helperFun.stopped(qd) and (np.linalg.norm(q - qgoal) < 0.1) or \





def _testPDControl():
    q = np.array([ 2.16254676,  0.02416342,  0.45212503, -0.77454544,  1.31134616,
        1.17771353,  0.12212003])
    qd = np.array([ 0.37019735, -0.39537201,  2.14345209,  0.05420211,  0.12789964,
        0.54933228, -1.31369193])
    qgoal = np.array([-3.61241883, -3.00785951,  1.24216599, -1.37947134, -6.27855665,
       -2.73050321,  1.57543535])
    print(PDControl(q, qd, qgoal))
