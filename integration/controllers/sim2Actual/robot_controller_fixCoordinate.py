from ikpy.chain import Chain
from controller import Supervisor
import numpy as np
import time
# from controller import Supervisor
#
# supervisor = Supervisor()
# timeStep = int(4 * supervisor.getBasicTimeStep())

class DRV90_Robot:
    def __init__(self, robot_urdf_name_down, end_down = True):
        self.motors = []
        self.armChain_down = Chain.from_urdf_file(robot_urdf_name_down)
        self.end_down = end_down
        self.supervisor = Supervisor()
        self.timestep = int(4 * self.supervisor.getBasicTimeStep())
        self.arm = self.supervisor.getSelf()
        self.armPosition = self.arm.getPosition()
        self.gripper_size = 0#0.1 # 0.2
        self.end_size = 0.0844695433179703 # 0.0845

        # self.armPosition = self.supervisor.getSelf.getPosition()

        for jointName in ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']:
            self.motor = self.supervisor.getDevice(jointName)
            self.motors.append(self.motor)
            self.position_sensor = self.motor.getPositionSensor()
            self.position_sensor.enable(self.timestep)

    def find_endposition(self):
        initial_position = [0] + [m.getPositionSensor().getValue() for m in self.motors]
        # initial_position = [0] + [m.getTargetPosition() for m in self.motors]
        # print('real',np.array(initial_position)/np.pi*180)
        if self.end_down:
            position = self.armChain_down.forward_kinematics(initial_position[:6])
            return [position[0, 3] - self.armPosition[0],
                    self.armPosition[1] - position[1, 3] - self.end_size - self.gripper_size,
                    self.armPosition[2] - position[2, 3]]
        else:
            position = self.armChain_down.forward_kinematics(initial_position)
            return [position[0, 3] - self.armPosition[0],
                    self.armPosition[1] - position[1, 3] - self.gripper_size,
                    self.armPosition[2] - position[2, 3]]



    def position_move(self, x_target, y_target, z_target):

        if self.end_down :
            x =    x_target - self.armPosition[0]
            y = - (y_target - self.armPosition[1]) - self.gripper_size - self.end_size
            z = - (z_target - self.armPosition[2])
            ikResults = self.armChain_down.inverse_kinematics([x, y, z])
            # xi, yi, zi = self.ideaPos(ikResults)
            ikResults[5] =  - np.pi / 2 + (ikResults[2] + ikResults[3])
            ikResults = np.append(ikResults, ikResults[1])
            # print(x,y, z)
            # print(ikResults)
            # print(self.armChain_down)
            # print('set',(ikResults) )
            for i in range(len(self.motors)):
                self.motors[i].setPosition(ikResults[i + 1])
            # return self.complete_check(x, y, z)
            return self.complete_check(x_target, y_target, z_target)
        else:
            x =    x_target - self.armPosition[0]
            y = - (y_target - self.armPosition[1]) - self.gripper_size
            z = - (z_target - self.armPosition[2])
            ikResults = self.armChain_down.inverse_kinematics([x, y, z])
            # xi, yi, zi = self.ideaPos(ikResults)
            for i in range(len(self.motors)):
                self.motors[i].setPosition(ikResults[i + 1])
            # return self.complete_check(x, y, z)
            return self.complete_check(x_target, y_target, z_target)

    def arm_joint_move(self, angle, axis='all'):
        if axis == 'all':
            position = self.ideaPos(angle)
            joint_move = np.array([0] + angle) * np.pi / 180.0
            # print(joint_move, position)
            for i in range(len(self.motors)):
                self.motors[i].setPosition(joint_move[i+1])
            return self.complete_check(position[0], position[1], position[2])

        elif axis >= 1 and axis <= 6:
            # joint_move = [0] + [m.getTargetPosition() for m in self.motors]
            joint_move = [m.getPositionSensor().getValue() for m in self.motors]
            joint_move[axis-1] = angle * np.pi / 180
            self.motors[axis-1].setPosition(joint_move[axis-1])
            position = self.ideaPos(joint_move)
            return self.complete_check(position[0], position[1], position[2])
        else:
            print("axis setting error!")

    def complete_check(self, x_target, y_target, z_target, iter_time = 0, iter_limit = 100):
        while iter_time < iter_limit and self.supervisor.step(self.timestep) != -1:
            x_f, y_f, z_f = self.find_endposition()
            diff = abs(x_f - x_target) + abs(y_f - y_target) + abs(z_f - z_target)
            iter_time += 1
            # print(x_f, y_f, z_f)
            # print(x_target, y_target, z_target)
            if abs(diff) < 1e-4:
                # print("True, iter", iter_time)
                return 0
        # print("False, diff", diff)
        print(diff, abs(x_f - x_target), abs(y_f - y_target), abs(z_f - z_target))
        return 1

    def currAngles(self):
        pos = []
        for m in self.motors:
            pos.append(m.getPositionSensor().getValue() / np.pi * 180.0)
        # initial_position = [0] + [m.getPositionSensor().getValue() for m in self.motors]
        return pos

    def ideaPos(self, angle):
        angle = np.array([0] + angle) * np.pi / 180.0
        position = self.armChain_down.forward_kinematics(angle)
        if self.end_down:
            return [position[0, 3], self.armPosition[1] - position[1, 3] - self.end_size - self.gripper_size,
                    self.armPosition[2] - position[2, 3]]
        else:
            return [position[0, 3], self.armPosition[1] - position[1, 3] - self.gripper_size,
                    self.armPosition[2] - position[2, 3]]
