import time
import torch
import math
import torchvision.transforms as transforms
import numpy as np
import time
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from ur_ikfast import ur_kinematics
from scipy.spatial.transform import Rotation as R
import tkinter as tk
import numpy as np
import time
# import sys, os
# sys.path.append('/home/mert/UR10_dockerized/scripts/')
# print (sys.path)
# import gamepad_control

rospy.init_node('gui', anonymous=True)
pub = rospy.Publisher('radian', Float64MultiArray, queue_size=10)

joint_order = [2, 1, 0, 3, 4, 5]
ur10_arm = ur_kinematics.URKinematics('ur10')

tool_length = 0.2 #distance from ur10 end and tool center

TASKSPACE_LIMITS = {'x':[-1.29, -0.62],'y':[-0.35, 0.6],'z':[0.24, 0.7]} # very very safe ;)
MAX_LIN_SPEED_TOOL = 0.05 # m/s
ROS_INTERFACE_RATE = 10 # Hz

dt = 1/ROS_INTERFACE_RATE
rate = rospy.Rate(ROS_INTERFACE_RATE) # 10hz   
def collect(current_state, current_pose):
    output_file = open("traj.txt", 'a')
    output_file.write(str(current_state)+'\n')
    output_file.flush()
    rate.sleep()
    output_file.close()
def return_to_pos():
    input_file = open("traj.txt", 'r')
    # get initial state from the file:
    traj = []
    for state in input_file:
        state= [float(i) for i in state[1:-2].split(',')]
        state = [state[i] for i in joint_order]
        traj.append(state)
    # execute trajectory in reverse order
    for state in reversed(traj):
        pub.publish(Float64MultiArray(data=state+[dt]))
        rate.sleep()
    input_file.close()
    open('traj.txt', 'w').close()  ## clear file 
    
def init_position():
    open('traj.txt', 'w').close()  ## clear file 
    input_file = open('init.txt', 'r')
    # get initial state from the file:
    current_state, _ = get_current_state(return_pose=True,collect_traj=False)
    print(current_state)
    starting_position = [-0.0008853117572229507, -1.5727275053607386, 0.0065135955810546875, -1.5761678854571741, -0.0015085379229944351, 0.0006707421271130443]
    for i in range(len(current_state)):
        if i ==0 :continue
        if abs (current_state[i] -starting_position[i] ) > 0.01:
            while True:
                current_state, _ = get_current_state(return_pose=True,collect_traj=False)
                print(current_state)
                print("not in inital position!!!!!!!")
                time.sleep(3)
    print("moving to starting position...")
    time.sleep(3)
    traj = []
    for state in input_file:
        state= [float(i) for i in state[1:-2].split(',')]
        state = [state[i] for i in joint_order]
        traj.append(state)
    for state in traj:
        pub.publish(Float64MultiArray(data=state+[dt]))
        rate.sleep()
    time.sleep(3)
    input_file.close()
def get_current_state(return_pose=True,collect_traj=True):
    current_state = rospy.wait_for_message("/joint_states", JointState)
    current_state_lst = [current_state.position[i] for i in joint_order]
    pose_matrix=None
    if return_pose:
        pose_matrix= ur10_arm.forward(current_state_lst, 'matrix')
    if collect_traj :
        collect(current_state_lst,pose_matrix )
    return current_state_lst, pose_matrix
##dx=0.01
def MoveArm(target_pose, current_pose=None, current_state=None, dx=0.008, time_from_start=1,secondTry=False):
    # theta target is ['x','y','Z'] first two are extrinsic euler angles, last one is intrinsic
    # target_pose and current_pose are 3x4 matrices !!!
    print("moving ... ")
    dtheta=(1/2*np.pi)/time_from_start

    if current_pose is None or current_state is None:
        current_state, current_pose = get_current_state(return_pose=True,collect_traj =True)

    p_initial = current_pose[:3,-1]
    p_target = target_pose[:3,-1]
    target_distance = np.linalg.norm(p_target-p_initial)
    target_speed = target_distance/time_from_start  
    
    # safety check - if speed demand is within limits
    if target_speed>MAX_LIN_SPEED_TOOL:
        return
    
    n_steps=max(2, int(abs(target_distance)/dx)+1)

    # dont allow to go beyond task space limits!

    for idx, axis in enumerate(['x','y','z']):
        p_target[idx] = np.clip(p_target[idx], TASKSPACE_LIMITS[axis][0], TASKSPACE_LIMITS[axis][1])
    #print("p_initial :  ", p_initial)
    #print("p_target  :  ", p_target)
    #print()

    p_vals = np.linspace(p_initial, p_target, n_steps)[1:]

    # get euler angles from pose matrix
    current_euler = R.from_matrix(current_pose[:3,:3]).as_euler('xyz', degrees=False)
    target_euler = R.from_matrix(target_pose[:3,:3]).as_euler('xyz', degrees=False)
    idx_max_angle_change = np.argmax(np.abs(target_euler-current_euler))
    n_steps2 = max(2, int(abs(target_euler[idx_max_angle_change]-current_euler[idx_max_angle_change])/dtheta)+1)
    n_steps = max(n_steps, n_steps2)
    
    int_eulers = np.linspace(current_euler, target_euler, n_steps)[1:] # intermediate euler angles
    p_vals = np.linspace(p_initial, p_target, n_steps)[1:]

    # rospy.loginfo("intermediate_p_vals:"+np.array2string(p_vals))
    data_to_send = Float64MultiArray()

    new_pose = current_pose
    for p_target_i, euler_i in zip(p_vals, int_eulers):
        new_pose[:3,-1] = p_target_i
        new_pose[:3,:3] = R.from_euler('xyz', euler_i, degrees=False).as_matrix()
        new_joint_pos = ur10_arm.inverse(new_pose, False, q_guess=current_state)
        if new_joint_pos is not None:
            # rospy.loginfo("going to "+np.array2string(new_pose))
            data_to_send.data = [new_joint_pos[i] for i in joint_order]+[time_from_start/n_steps]
            pub.publish(data_to_send)
        else:
            rospy.loginfo("cant find solution for the orientation:"+np.array2string(euler_i))
    
    time.sleep(1)
    ## wait until target is reached
    current_state, current_pose = get_current_state(return_pose=True,collect_traj=False)
    p_initial = current_pose[:3,-1]
    if (secondTry==False and (abs(p_initial[0] - p_target[0]) > 0.0015 or abs(p_initial[1] - p_target[1]) > 0.0015 )):# ee pos dacozebılırsın bunu
        
        #print(abs(p_initial[0] - p_target[0]) )
        #print(abs(p_initial[1] - p_target[1]))        
        time.sleep(1)
        MoveArm(target_pose, current_pose=None, current_state=None, dx=0.008, time_from_start=1,secondTry=True)

        
 
    
# todo: guarantee that speed is not higher when going in diagonal direction!
def compute_target_pose(current_pose, v, w_x, w_y, w_Z, dt):
    # compute the target position
    p_initial = current_pose[:3,-1]
    target_speed = (np.linalg.norm(v)/np.sqrt(3.0))*MAX_LIN_SPEED_TOOL
    target_distance = target_speed*dt
    p_target = p_initial + target_distance*(v/(np.linalg.norm(v)+1e-5))
    # compute the target orientation
    R_initial = current_pose[:3,:3]
    R_target = R_initial
    R_target = np.dot(R_target, R.from_euler('x', w_x*dt, degrees=True).as_matrix())
    R_target = np.dot(R_target, R.from_euler('y', w_y*dt, degrees=True).as_matrix())
    R_target = np.dot(R.from_euler('Z', w_Z*dt, degrees=True).as_matrix(), R_target)
    target_pose = np.zeros((3,4), dtype=np.float32)
    target_pose[:3,:3] = R_target
    target_pose[:,-1] = p_target
    return target_pose
## action :     [0.05 0.  ]
## action :     [-0.025       0.04330127]
## action :     [-0.025      -0.04330127]
## action :     [ 5.0000000e-02 -1.2246468e-17]
def convert_to_speed (x, y): ## convert cartesian to speed  -divide if it exceeds 0.3 or 1 !
### v =             0.03 --- >1
### max action val  0.067 --->
    signX = 1
    signY = 1
    if x <0 :
        signX = -1
    if y <0 :
        signY = -1 
    print("values:  " , x,y)
    x = abs (x*33.333)   ## convert cartesian to speed
    y = abs (y*33.333)
    
    lenght = max(math.ceil(abs(x)) ,math.ceil(abs(y)) )
    x_speed =np.zeros(lenght)   
    y_speed =np.zeros(lenght)
    for i in range((lenght)):
        if x -1.0 > 0:
            x_speed[i] = 1.0 * signX
            x = x - 1.0
        else : 
            x_speed[i] = x * signX
            break
    for i in range((lenght)):
        if y-1.0 > 0:
            y_speed[i] = 1.0 * signY
            y = y - 1.0
        else : 
            y_speed[i] = y * signY
            break
    if len(x_speed) !=3 or len(y_speed)!=3:
        print("ERROR - check speeds!")
        print(x_speed)
        print(y_speed)
        x_speed =r=np.zeros(lenght)   
        y_speed =r=np.zeros(lenght)
        while(True):
            time.sleep(3)
    print("x_speed:  ", x_speed)
    print("y_speed:  ", y_speed)
    return x_speed , y_speed

def convert_sim_to_real (x, y):
    simRangeX = 0.5
    simRangeY = 0.6  
    realRangeX = 0.67 
    realRangeY = 0.95
    r=np.zeros(2)
    r[0]=((x * realRangeX) / simRangeX)  
    r[1]=((y * realRangeY) / simRangeY) 
    return r
def convert_real_to_sim (x, y):  ## cartesian conversion for sim to real 
    simRangeX = 0.5
    simRangeY = 0.6
    realRangeX = 0.67 
    realRangeY = 0.95
    r=np.zeros(2)
    r[0]=((x * simRangeX) / realRangeX)  
    r[1]=((y * simRangeY) / realRangeY) 
    return r

#### butun valuelar sım e gore olacak!!!!!
v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
w_Z = 0.0
w_x = 0.0
w_y = 0.0
dT  = 1.0
class Env():
    def __init__(self):
        #rospy.init_node('gui', anonymous=True)

        self._t = 0
        self._goal_thresh = 0.01
        self._max_timesteps = 50
        self.n_actions = 4
        self._delta = 0.05
        theta = np.linspace(0, 2*np.pi, self.n_actions)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}
        self.init_goal_pos()##self var mı burda bak 
        init_position() 

    def reset(self, seed=None):
        self._t = 0
        return_to_pos() 
        self.init_goal_pos()
        


    def init_goal_pos(self):
        x = np.random.uniform(0.25, 0.75)
        y =   np.random.uniform(-0.3, 0.3)
        self.goal_pos = [x, y, 0.6]  
        

        

    def step(self, action_id):
        # if ( rospy.is_shutdown()):
        #     return
        action = self._actions[action_id] * self._delta
        action = convert_sim_to_real(action[0],action[1])
        v_x , v_y = convert_to_speed(action[0],action[1])
        for i in range(3):  ## move 
            v = np.zeros(3)
            v[0]=v_x[i]
            v[1]=v_y[i]
            v[2]=0.0
            current_state, current_pose  =   get_current_state(return_pose=True)
            target_pose = compute_target_pose(current_pose, v, w_x, w_y, w_Z, dt=dT)
            MoveArm(target_pose, current_state=current_state, current_pose=current_pose, time_from_start=dT,secondTry=False)
            rate.sleep()
        print("step:  "  , self._t)
        self._t = self._t + 1  
        
        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        
        return state, reward, terminal, truncated
         

    def high_level_state(self):
        _, current_pose  =   get_current_state(return_pose=True)
        p_initial = current_pose[:3,-1]
        self.ee_pos = convert_real_to_sim (p_initial[0],p_initial[1])
        return np.concatenate([self.ee_pos[:2], self.goal_pos[:2]])

    def reward(self):
        ee_to_goal = max(100*np.linalg.norm(self.ee_pos[:2] - self.goal_pos[:2]), 1)
        print ("ee pos :   ", self.ee_pos[:2])
        print("goal pos :  ", self.goal_pos[:2])
        print("reward:   ", 1/(ee_to_goal))
        return 1/(ee_to_goal) 

    def is_terminal(self):
        return np.linalg.norm( self.goal_pos[:2] - self.ee_pos[:2]) < self._goal_thresh  ## bunlar sıme gore mı real gore mı bak 

    def is_truncated(self):
        return self._t >= self._max_timesteps