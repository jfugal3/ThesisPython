import controllers
import data_calc
import randVelGen
import bounds
import helperFun
import sys
import csv
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState

VEL_MAG = 1.0
RETURN_TIME = 0.5
NUM_TRAJECTORIES = 500

ACCELERATING = 0
COASTING = 1
DAMPING = 2
COMING_BACK_IN_BOUNDS = 3
FINISHED = 4

q_INDEX = 1
qd_INDEX = 2

TIMESTEP = 0.002

def print_count(count):
    cur = str(count)
    prev = str(count - 1)
    for c in prev:
        sys.stdout.write("\b")
    sys.stdout.write(cur)
    sys.stdout.flush()

def main(data_dir):
    model = load_model_from_path("robot.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    initial_state = MjSimState(
        time=0,
        qpos=np.array([0, -np.pi/4, 0, -3 * np.pi/4 + 1, 0, np.pi/2, np.pi/4]),
        qvel=np.zeros(7),
        act=None,
        udd_state={})
    sim.set_state(initial_state)
    sim.step()
    traj_count = 0
    control_state = None
    while(traj_count < NUM_TRAJECTORIES and control_state != FINISHED):
        control_state = ACCELERATING
        outputFile = None
        initial_q = sim.get_state()[q_INDEX]
        velGen = randVelGen.RandVelGenerator()
        qd_des = velGen.generatePatternVel()
        coming_back_time = 0.0
        time = 0
        while control_state != FINISHED:
            state = sim.get_state()
            q = state[q_INDEX]
            qd = state[qd_INDEX]
            boundViolations = bounds.getBoundViolations(q)
            # RD = 
            TB = bounds.tableBoundViolation(sim)
            OB = bounds.outOfBounds(q)
            DA = helperFun.moving(qd)
            DC = helperFun.stopped(qd)
            DD = DC
            DB = coming_back_time > RETURN_TIME
            FN = traj_count >= NUM_TRAJECTORIES

            prev_state = control_state
            # transition block
            if control_state == ACCELERATING:
                if not TB and not OB and not DA:
                    control_state = ACCELERATING
                elif TB:
                    control_state = COMING_BACK_IN_BOUNDS
                    coming_back_time = 0.0
                elif not TB and OB:
                    control_state = DAMPING;
                    curBoundViolations = bounds.getBoundViolations(q)
                    velGen.setJointDirections(curBoundViolations)
                elif not TB and not OB and DA:
                    control_state = COASTING
                    traj_count += 1
                    outputFile = open(data_dir + helperFun.getUniqueFileName("traj"), mode='x')
                    outputWriter = csv.writer(outputFile, delimiter=',')
                    print_count(traj_count)
                else:
                    control_state = FINISHED
                    print("Unknown transistion! ACCELERATING")

            elif control_state == COASTING:
                if not FN and not TB and not OB and DC:
                    control_state = ACCELERATING
                    qd_des = velGen.generatePatternVel()
                    outputFile.close()
                elif not FN and TB:
                    control_state = COMING_BACK_IN_BOUNDS
                    coming_back_time = 0
                    outputFile.close()
                elif not FN and not TB and OB:
                    control_state = DAMPING
                    outputFile.close()
                    curBoundViolations = bounds.getBoundViolations(q)
                    velGen.setJointDirections(curBoundViolations)
                elif FN:
                    control_state = FINISHED
                    outputFile.close()
                elif not FN and not TB and not OB and not DC:
                    control_state = COASTING
                else:
                    control_state = FINISHED
                    print("Unknown transition! COASTING")
                    outputFile.close()

            elif control_state == DAMPING:
                if not TB and not DD:
                    control_state = DAMPING
                elif TB:
                    control_state = COMING_BACK_IN_BOUNDS
                    coming_back_time = 0.0
                elif not TB and DD:
                    control_state = ACCELERATING
                    qd_des = velGen.generatePatternVel()
                else:
                    control_state = FINISHED
                    print("Unknow transition! DAMPING")

            elif control_state == COMING_BACK_IN_BOUNDS:
                if not DB:
                    control_state = COMING_BACK_IN_BOUNDS
                elif DB and OB:
                    control_state = DAMPING
                    curBoundViolations = bounds.getBoundViolations(q)
                    velGen.setJointDirections(curBoundViolations)
                elif DB and not OB:
                    control_state = ACCELERATING
                    qd_des = velGen.generatePatternVel()
                else:
                    control_state = FINISHED
                    print("Unknown transition! COMING_BACK_IN_BOUNDS")

            elif control_state == FINISHED:
                control_state = FINISHED

            else:
                control_state = FINISHED
                print("Got to an invalid state!")

            # debug states
            if prev_state != control_state:
                if control_state == ACCELERATING:
                    print("ACCELERATING")
                elif control_state == COASTING:
                    print("COASTING")
                elif control_state == DAMPING:
                    print("DAMPING")
                elif control_state == COMING_BACK_IN_BOUNDS:
                    print("COMING_BACK_IN_BOUNDS")
                elif control_state == "FINISHED":
                    print("FINISHED")
                else:
                    print("In a bad state!")

            torques = np.zeros(7)
            if control_state == ACCELERATING:
                torques = controllers.basicVelControl(qd_des=qd_des, qd_cur=qd)

            elif control_state == COASTING:
                data = np.concatenate((q,qd,data_calc.get_3D_data(sim),[time]))
                outputWriter.writerow(data)
                torques = controllers.basicVelControl(qd_des=qd_des, qd_cur=qd)

            elif control_state == DAMPING:
                torques = controllers.dampingControl(qd)

            elif control_state == COMING_BACK_IN_BOUNDS:
                coming_back_time += TIMESTEP
                torques = controllers.moveAwayFromTable(q=q,qd=qd)

            elif control_state == FINISHED:
                outputFile.close()
                break
            else:
                print("Got to an invalid state!")
                control_state = FINISHED
                break

            sim.data.ctrl[:] = torques
            sim.step()
            viewer.render()
            time += TIMESTEP

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:",sys.argv[0],"<data directory>")
    else:
        print("calling main")
        main(sys.argv[1])
