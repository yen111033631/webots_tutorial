import cv2
import random
import numpy as np
from ikpy.chain import Chain
from gymnasium import spaces
from controller import Supervisor

#------------------------------------------------------------------
# set random seed 
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

setup_seed(525)
#------------------------------------------------------------------
# set arm communication with webots
class DRV90_Robot:
    def __init__(self, robot_urdf_name_down, Super, end_down = True):
        self.motors = []
        self.armChain_down = Chain.from_urdf_file(robot_urdf_name_down)
        self.end_down = end_down
        self.supervisor = Super
        self.timestep = int(4 * self.supervisor.getBasicTimeStep())
        self.arm = self.supervisor.getSelf()
        self.armPosition = self.arm.getPosition()
        self.gripper_size = 0 #0.1 # 0.2
        self.end_size = 0.0844695433179703 # 0.0845
        
        # get boundary of the arm with angle and radian
        self.get_chain_bounds()

        # print(self.armChain_down.links)
        # print(self.armChain_down.name)
        # self.armPosition = self.supervisor.getSelf.getPosition()

        for jointName in ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']:
            self.motor = self.supervisor.getDevice(jointName)
            self.motors.append(self.motor)
            self.position_sensor = self.motor.getPositionSensor()
            self.position_sensor.enable(self.timestep)
    
    def get_chain_bounds(self, restrict_angle=3):
        def restrict_bound(bound, num):
            # restrict the link boundary
            result = [None, None]
            result[0] = bound[0] + num
            result[1] = bound[1] - num
            return np.asarray(result)
        
        self.links_bounds_angle = []
        self.links_bounds_radian = []
        for i in range(len(self.armChain_down) - 1):
            bound_radian = np.asarray(self.armChain_down.links[i+1].bounds) 
            bound_radian = restrict_bound(bound_radian, restrict_angle * np.pi / 180)
            self.links_bounds_radian.append(bound_radian)
            self.links_bounds_angle.append(bound_radian * 180.0 / np.pi)

    def check_ik_in_boundarys(self, ik):
        """
        bounds and ik should be same type (angle or radian)
        """
        is_ok = True
        assert len(self.links_bounds_angle) == len(ik), "len(self.links_bounds_angle) should == len(ik)"
        for i, bound in enumerate(self.links_bounds_angle):
            if not(bound[0] < ik[i] < bound[1]):
                is_ok = False
                break
            else:
                pass
        return is_ok

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
                    position[2, 3] - self.armPosition[2],
                    -(position[1, 3] - self.armPosition[1]) - self.gripper_size]

    def position_move(self, x_target, y_target, z_target):

        if self.end_down :
            x =    x_target - self.armPosition[0]
            y = - (y_target - self.armPosition[1]) - self.gripper_size - self.end_size
            z = - (z_target - self.armPosition[2])
            ikResults = self.armChain_down.inverse_kinematics([x, y, z])
            # xi, yi, zi = self.ideaPos(ikResults)
            ikResults[5] =  - np.pi / 2 + (ikResults[2] + ikResults[3])
            ikResults = np.append(ikResults, ikResults[1])
            # print("123",x,y,z)
            # print(ikResults)
            # print(self.armChain_down)
            # print('set',(ikResults) )
            for i in range(len(self.motors)):
                self.motors[i].setPosition(ikResults[i + 1])
            # return self.complete_check(x, y, z)
            return self.complete_check(x_target, y_target, z_target)
        else:
            x =  (x_target - self.armPosition[0])
            y = -(z_target - self.armPosition[2])
            z = (y_target - self.armPosition[1]) + self.gripper_size
            ikResults = self.armChain_down.inverse_kinematics([x, y, z])
            # xi, yi, zi = self.ideaPos(ikResults)
            for i in range(len(self.motors)):
                self.motors[i].setPosition(ikResults[i + 1])
            # return self.complete_check(x, y, z)
            return self.complete_check(x_target, y_target, z_target)

    def arm_joint_move(self, angle, axis='all'):
        if axis == 'all':
            joint_move = np.array([0] + angle) * np.pi / 180.0
            # print(joint_move, position)
            for i in range(len(self.motors)):
                self.motors[i].setPosition(joint_move[i+1])
                # print(joint_move[i+1])
            position = self.ideaPos(angle)
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
            if abs(diff) < 1e-4:
                return 0
        print("complete check",diff, abs(x_f - x_target), abs(y_f - y_target), abs(z_f - z_target))
        return 1
        
    def currAngles(self):
        pos = []
        for m in self.motors:
            pos.append(m.getPositionSensor().getValue() * 180.0 / np.pi )
        return pos

    def ideaPos(self, angle):
        angle = np.array([0] + angle) * np.pi / 180.0
        position = self.armChain_down.forward_kinematics(angle)
        if self.end_down:
            return [position[0, 3] - self.armPosition[0],
                    self.armPosition[1] - position[1, 3] - self.end_size - self.gripper_size,
                    self.armPosition[2] - position[2, 3]]
        else:
            return [position[0, 3] - self.armPosition[0],
                    position[2, 3] - self.armPosition[2],
                    -(position[1, 3] - self.armPosition[1]) - self.gripper_size]


    def movJ(self, setJD):
        while self.angleCompare(setJD) >= 2 and self.supervisor.step(self.timestep) != 1:
            self.setMotor(setJD)
            # print(self.angleCompare(setJD))
        return self.checkJ(setJD[:], setJD[:])

    def movP(self, target_position, target_orientation=[0, 1, 0], orientation_axis="Z"):

        ikAnglesD = self.get_IK_angle(target_position=target_position,
                                      target_orientation=target_orientation,
                                      orientation_axis=orientation_axis)

        return self.movJ(ikAnglesD)
    
    ############################################################################################
    def setP(self, target_position, target_orientation=[0, 1, 0], orientation_axis="Z"):
        ikAnglesD = self.get_IK_angle(target_position=target_position,
                                      target_orientation=target_orientation,
                                      orientation_axis=orientation_axis)
        
        return 0
    ############################################################################################

    
    def get_IK_angle(self, target_position, target_orientation=[0, 1, 0], orientation_axis="Z"):
        x = (target_position[0] - self.armPosition[0])
        y = (self.armPosition[2] - target_position[2])
        z = (target_position[1] - self.armPosition[1])
        ikAnglesR = self.armChain_down.inverse_kinematics(
            target_position = [x,y,z], 
            target_orientation=target_orientation, 
            orientation_mode=orientation_axis)
        ikAnglesD = (ikAnglesR[1:] * 180.0 / np.pi).tolist()
        return ikAnglesD

    def getCurJ(self):
        '''
        :return:Current Joint Angle in Degrees
        '''
        curJ = []
        for m in self.motors:
            curJ.append(m.getPositionSensor().getValue() * 180.0 / np.pi )
        return curJ

    def getCurP(self):
        curJ = self.getCurJ()
        curJR = np.array([0]+curJ) / 180.0 * np.pi
        fkMatrix = self.armChain_down.forward_kinematics(curJR)
        return [fkMatrix[0,3], fkMatrix[2,3], -fkMatrix[1,3]+self.armPosition[2]]

    def getCurP_Joint(self, which_joint):
        curJ = self.getCurJ()
        curJR = np.array([0]+curJ) / 180.0 * np.pi
        fkMatrix_all = self.armChain_down.forward_kinematics(curJR, full_kinematics=True)
        fkMatrix = fkMatrix_all[which_joint]
        
        return [fkMatrix[0,3], fkMatrix[2,3], -fkMatrix[1,3]+self.armPosition[2]]

    def checkJ(self, compJ, setJ, iterTime = 0, iterLimit = 50):
        compJ = self.compSetJ(compJ, setJ)
        self.setMotor(compJ)
        errD  = self.angleCompare(setJ)
        if (iterTime + 1) > iterLimit:
             print("Check J : ",iterTime, errD, compJ)
             return 1
        if self.supervisor.step(self.timestep) != 1:
            if errD < 1e-2:
                '''
                errD:1e-1 need 03 iterTimes
                errD:1e-2 need 03 - 34 iterTimes
                errD:2e-3 need 47 - 60 iterTimes
                errD:1.7e-3 need 48 - 75 iterTimes
                '''
                # print(iterTime, errD)
                return 0
            else:
                return self.checkJ(compJ, setJ, iterTime + 1)


    def setMotor(self, setJ, rad = False):
        '''
        :param setJ: Motor Setting Angles
        :param rad: Does Input Angle in Radius(Default: False)
        :return: Current Angle in degrees
        '''
        if not rad:
            setJR = np.array([0] + setJ) / 180.0 * np.pi
        else:
            setJR = setJ

        for i in range(len(self.motors)):
            self.motors[i].setPosition(setJR[i+1])
        curJ = self.getCurJ()
        return curJ

    def compSetJ(self, compJ, setJ):
        '''
        :param compJ: Previous Compensated Angle
        :param setJ: Settign Angles
        :return: Compensated Angles
        '''
        curJ = self.getCurJ()
        compCoef = 0.1
        for i in range(len(curJ)):
            compJ[i] = compJ[i] + (setJ[i] - curJ[i]) * compCoef
        # print("com", compJ)
        # print("set", setJ)
        # print("cur",curJ,'\n')
        # print("set", setJ)
        return compJ

    def angleCompare(self, setJ):
        '''
        :param setJ: Setting Angles in Degrees
        :return: Error between settingAngles and CurrentAngles in Degrees
        '''
        curJ = self.getCurJ()
        errD  = 0
        for i in range(len(curJ)):
            errD += abs(setJ[i] - curJ[i])
        return errD


def unzip_list(key_list):
    result = []
    for sublist in key_list:
        result.extend(sublist)
    
    return result

#------------------------------------------------------------------
# set env 
class env_ball:
    def __init__(self, seed, state_type="numerical", object_name="ball", is_going_to_target=False, big_negative_reward=-500, reward_slope=-10):
        """_summary_

        Args:
            seed (_type_): _description_
            state_type (str, optional): _description_. Defaults to "numerical". ["numerical", "RGB", "gray"]
            object_name (str, optional): _description_. Defaults to "ball".  ["ball", "cube"]
            is_going_to_target (bool, optional): _description_. Defaults to False.
            big_negative_reward (int, optional): _description_. Defaults to -500.
            reward_slope (int, optional): _description_. Defaults to -10.
        """
        
        assert type(seed) == int and type(state_type) == str, "fuck you"
        #------------------------------------------------------------------------------
        # some init setting
        self.seed = seed
        self.supervisor = Supervisor()
        self.observation = observation()
        self.info = f"{object_name}_{state_type}" 
        self.max_num_step = 100
        self.constant_displacement = 0.05
        self.env_bounds = [[-0.27, 0.48],
                          [-0.74, 0.74],
                          [0.04, 1]
                          ]        
        #------------------------------------------------------------------------------
        # reward
        self.big_negative_reward = big_negative_reward
        self.reward_slope = reward_slope        
        #------------------------------------------------------------------------------
        # object
        self.ball = Object(object_name, self.supervisor)
        self.arm = DRV90_Robot("DRV90.urdf", end_down=False, Super=self.supervisor)
        self.cam = Camera("camera", self.supervisor, self.arm.timestep)
        #------------------------------------------------------------------------------
        # state and action space 
        self.state_type = state_type
        self.action_space = spaces.Discrete(6)
        
        if self.state_type == "numerical":
            self.observation_space = 10
        else:
            channel_n = 3 if  self.state_type == "RGB" else 1
            self.observation_space = spaces.Box(low=0, 
                                                high=255,
                                                shape=(channel_n, 84, 84), 
                                                dtype=np.uint8)
            
        # action type  
        self.is_going_to_target = is_going_to_target
        #------------------------------------------------------------------------------
        # print env info 
        print("==" * 10)
        print(f"env info:\t\t{self.info}")
        print(f"observation_shape:\t{self.observation_space}")
        print(f"action_space:\t{self.action_space}")
        print("==" * 10)
        #------------------------------------------------------------------------------
        

    def reset(self, seed=0, is_random=True):
        #------------------------------------------------------------------------------
        # init env seed
        random.seed(self.seed + seed)      
        #------------------------------------------------------------------------------
        # is target object random or not 
        if is_random:
            self.ball.random_choose_one_position()
        else:
            self.ball.new_position = [0.37, 0.54, 0.09]
        #------------------------------------------------------------------------------
        # set target object
        self.ball.setP(self.ball.new_position)        
        #------------------------------------------------------------------------------
        # init arm
        arm_start_position = [0.3, 0, 0.55]
        self.arm.movP(arm_start_position)
        #------------------------------------------------------------------------------
        # set observation and state
        self.get_observation()
        self.get_state()
        #------------------------------------------------------------------------------
        # reset step 
        self.num_step = 0
        #------------------------------------------------------------------------------
        return self.state, self.info
    
    def get_observation(self):
        #------------------------------------------------------------------------------
        # set arm observation
        self.observation.arm = {"joint_angle" : np.asarray(self.arm.getCurJ()),
                                "end_point" : np.asarray(self.arm.getCurP())}
        #------------------------------------------------------------------------------
        # set object observation
        self.observation.object_position = self.ball.position
        #------------------------------------------------------------------------------
        # move object_position above the object 
        self.observation.object_position[-1] = self.observation.object_position[-1] + 0.07
        #------------------------------------------------------------------------------
        # set cam observation
        self.frame = self.cam.get_frame()
        # self.cam.show_frame() # TODO add cam infomation
        #------------------------------------------------------------------------------
        return None
    
    def get_state(self):
        #------------------------------------------------------------------------------
        # set numerical state 
        if self.state_type == "numerical":
            ## joint angle: 6 (not used now)
            ## end point: 3
            ## ball position: 3
            ## relationship vector: 3
            ## relationship distance: 1
            ## total: 10
            
            # FIXME now is only arm value and ball value
            # state_list = list(self.observation.arm.values())
            state_list = list([self.observation.arm["end_point"]])
            state_list.append(self.observation.object_position)

            # relationship between arm end point and ball 
            vector = np.asarray(self.observation.object_position)-np.asarray(self.observation.arm["end_point"])
            self.calculate_distance()
            relationship = [*(vector.tolist()), self.distance]
            state_list.append(relationship)            

            self.state = unzip_list(state_list)
        #------------------------------------------------------------------------------
        # set RGB state (3 channels) 
        elif self.state_type == "RGB":
            # RGB
            self.state = self.frame[:, :, :3]        
        #------------------------------------------------------------------------------
        # set gray state (1 channels)
        elif self.state_type == "gray":
            # gray image
            self.state = cv2.cvtColor(self.frame[:, :, :3], cv2.COLOR_BGR2GRAY)            
            self.state = np.expand_dims(self.state, axis=-1)   
        #------------------------------------------------------------------------------
        return self.state

    def step(self, action):

        #------------------------------------------------------------------------------
        # execute action
        self.execute_action(action)
        #------------------------------------------------------------------------------
        # get next state
        self.get_observation()
        self.get_state()
        #------------------------------------------------------------------------------
        # calculate distance
        self.calculate_distance()
        #------------------------------------------------------------------------------
        # check if task is terminated or truncated
        terminated = False
        truncated = False
        
        # terminated
        if self.distance <= 0.1 or not(self.is_good):
            terminated = True

        # truncated
        if self.num_step >= self.max_num_step:
            truncated = True

        #------------------------------------------------------------------------------
        # calculate reward   
        if self.is_good:
            self.reward = self.distance * self.reward_slope  # FIXME reward design
        else:
            self.reward = self.big_negative_reward
        #------------------------------------------------------------------------------

        return self.state, self.reward, terminated, truncated, self.info
    
    def execute_action(self, action):
        """
        first one: discrete
        choose +-(x, y, z) direction with constant displacement
        """        

        #------------------------------------------------------------------------------
        # get now observation
        self.get_observation()
        #------------------------------------------------------------------------------
        # turn action into displacement        
        assert 0 <= action <= 5, "action index not between 0-5"
        displacement = 0
        if action == 0:
            displacement = np.asarray([self.constant_displacement, 0, 0])
        elif action == 1:
            displacement = np.asarray([-self.constant_displacement, 0, 0])
        elif action == 2:
            displacement = np.asarray([0, self.constant_displacement, 0])
        elif action == 3:
            displacement = np.asarray([0, -self.constant_displacement, 0])
        elif action == 4:
            displacement = np.asarray([0, 0, self.constant_displacement])
        elif action == 5:
            displacement = np.asarray([0, 0, -self.constant_displacement])
        
        #------------------------------------------------------------------------------
        # add action into next position
        now_arm_position = np.asarray(self.observation.arm["end_point"])
        next_position = now_arm_position + displacement
        
        #------------------------------------------------------------------------------
        # check position in boundary (both arm and env)
        ik = self.arm.get_IK_angle(next_position)
        
        is_ik_good = self.arm.check_ik_in_boundarys(ik)
        is_position_good = self.check_position_in_boundary(next_position)
        self.is_good = is_ik_good and is_position_good
        
        #------------------------------------------------------------------------------
        # execute action
        if self.is_going_to_target:
            self.arm.movP(self.observation.object_position)
        
        elif self.is_good:
            self.arm.movP(next_position)
            
        else:
            print("-" * 5)
            print("bad action")
            print("next_position", next_position)
            print("joint boundary (is_ik_good)", is_ik_good)
            print("env boundary (is_position_good)", is_position_good)
            print("-" * 5)
            pass
        #------------------------------------------------------------------------------
        # add 1 num_step
        self.num_step += 1
        #------------------------------------------------------------------------------
        return None
    
    def calculate_distance(self):
        #------------------------------------------------------------------------------
        # define a_position and b_position
        a_position = self.observation.arm["end_point"]
        b_position = self.observation.object_position

        assert len(a_position) == len(b_position), "Error: length not the same, cant calculate"
        #------------------------------------------------------------------------------
        # calculate distance
        tem = 0
        for i in range(len(a_position)):
            tem += (a_position[i] - b_position[i]) ** 2
        
        self.distance = tem ** 0.5
        #------------------------------------------------------------------------------
        return self.distance
    
    def check_position_in_boundary(self, position):
        """
        bounds and position should be same type (angle or radian)
        """
        #------------------------------------------------------------------------------
        # check lens  
        assert len(self.env_bounds) == len(position), "len(self.env_bounds) should == len(position)"
        #------------------------------------------------------------------------------
        # check env boundary
        is_ok = True
        for i, bound in enumerate(self.env_bounds):
            if not(bound[0] < position[i] < bound[1]):
                is_ok = False        
        #------------------------------------------------------------------------------
        # check arm third joint doesn't collide with back wall
        if is_ok:
            joint3_position = self.arm.getCurP_Joint(3)
            if joint3_position[0] <= self.env_bounds[0][0] + 0.13:
                is_ok = False
        #------------------------------------------------------------------------------
        return is_ok

class observation:
    def __init__(self) -> None:
        self.arm = None
        self.cam = None
        self.object_position = None

class Camera:
    def __init__(self, DEF_name, Super, timestep) -> None:
        self.name = DEF_name
        self.cam_node = Super.getDevice(DEF_name)
        self.cam_node.enable(timestep)
        self.frame = None
    
    def get_frame(self):
        self.frame = np.frombuffer(self.cam_node.getImage(), dtype=np.uint8).reshape((self.cam_node.getHeight(), self.cam_node.getWidth(), 4))
        return self.frame
    
    def show_frame(self):
        cv2.imshow("cam_node image", self.get_frame())
        cv2.waitKey(1) 
        return None

class Object:
    def __init__(self, DEF_name, supervisor):
        self.name = DEF_name
        self.node = supervisor.getFromDef(DEF_name)
        self.translation_field = self.node.getField('translation')
        self.position = self.node.getPosition()
        
        if DEF_name == "ball":
            self.position_candididate = [[-0.23, 0.5, 0.24],
                                         [0.42, 0.5, 0.24],
                                         [0.02, 0.45, 0.86],
                                         [0.02, 0.35, 0.6],
                                         [0.02, -0.44095, 0.16704],
                                         [0.195, -0.22661, 0.38138],
                                         [0.37871, -0.62161, 0.62638],
                                         [0.20724, -0.57161, 0.71638]]
        elif DEF_name == "cube":
            self.position_candididate = [[0,    -0.55,  0.09],
                                         [0.4,  -0.3,   0.09],
                                         [0.31, -0.44,  0.09],
                                         [0.38, 0.07,   0.09],
                                         [0.31, 0.2,    0.09],
                                         [0.34, 0.32,   0.09],
                                         [0.01, 0.36,   0.09],
                                         [0.1,  0.63,   0.09],
                                         [0.37, 0.54,   0.09],
                                        ]
        self.new_position = None

    def get_position(self):
        self.position = self.node.getPosition()
        return self.position
    
    def random_choose_one_position(self):
        random_index = random.randint(0, len(self.position_candididate) - 1)
        self.new_position = self.position_candididate[random_index].copy()
        
        # --------------------------------------------------------------------------------------------------------------
        # TODO part absolute random 
        # if self.name == "ball" or self.name == "cube":
        # elif self.name == "cube":
        #     x_min, x_max = -0.16, 0.33
        #     y_min, y_max = -0.55, 0.55
        #     interval = 0.01

        #     x_min_notgood, x_max_notgood = -0.16, 0.3
        #     y_min_notgood, y_max_notgood = -0.27, 0.27

        #     # 生成抽樣點
        #     x_values = np.round(np.arange(x_min, x_max + interval, interval), 2)
        #     y_values = np.round(np.arange(y_min, y_max + interval, interval), 2)

        #     while True:
        #         # 隨機抽取一個值
        #         x_coor = np.random.choice(x_values)
        #         y_coor = np.random.choice(y_values)
                
        #         # check boundary
        #         if not((x_min_notgood <= x_coor <= x_max_notgood) and (y_min_notgood <= y_coor <= y_max_notgood)):
        #             break
            
        #     self.new_position = [x_coor, y_coor, 0.09]
        # --------------------------------------------------------------------------------------------------------------
            
        return self.new_position

    def setP(self, position):
            self.translation_field.setSFVec3f(position)
            self.position = position
            # self.get_position()

