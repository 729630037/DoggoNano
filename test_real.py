import os,sys
from math import pi as PI, degrees, radians, sin, cos,sqrt,pow,atan2,acos
import math
import threading
import queue
import signal
import time

from drivers.imu_BNO008X_i2c import IMU      
from drivers.position_control_threading import PositionControl
from envs.minitaur_trotting_env import MinitaurTrottingEnv
from envs.tools import bullet_client
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import suite_pybullet

def convert_to_tensor(time_step):
    step_type=time_step[0]
    reward=time_step[1]
    discount=time_step[2]
    observation=time_step[3]    
    step_type = tf.convert_to_tensor(
        [step_type], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor(
        [reward], dtype=tf.float32, name='reward')
    discount = tf.convert_to_tensor(
        [discount], dtype=tf.float32, name='discount')
    observation = tf.convert_to_tensor(
        [observation], dtype=tf.float32, name='observation')
    time_step = ts.TimeStep(step_type, reward, discount, observation)
    return time_step

imu=IMU()
saved_policy = tf.saved_model.load("policies/policy")
time_step=[0,0,0,imu.DataHandle()]
# env_name = "MinitaurTrottingEnv-v1"
# eval_env = suite_pybullet.load(env_name,max_episode_steps=2000)
# time_step= eval_env.reset()
# time_step=convert_to_tensor(time_step)
# reward=0
flag=True

def handler(signum, frame):
    global flag
    flag=False

signal.signal(signal.SIGINT,handler)
pos_control=PositionControl()
pos_control.Start()
while pos_control.ready!=[1]*4 :
    pass
t=input("please input t:")
while t!='t':
    pass
time_init=time.time()



while flag:
    action_step = saved_policy.action(time_step)
    proto_tensor=tf.make_tensor_proto(action_step.action)
    action=tf.make_ndarray(proto_tensor)
    action=[[0,0,0,0,0,0,0,0]]
    pos_control.TransformActionToMotorCommand(time.time()-time_init,action)  
    # time_step = eval_env.step(action[0])
    time_step=[0,0,0,imu.DataHandle()]
    # print(eval_env.action)

    time_step=convert_to_tensor(time_step)



