

r"""Running a pre-trained ppo agent on minitaur_trotting_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
if os.getcwd() not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append('/home/nano/minitaur-nano')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import tensorflow.compat.v1 as tf
from agents.scripts import utility
import pybullet_data
from envs import simple_ppo_agent
from envs.drivers import position_control_threading
from envs.drivers import imu_BNO008X_uart
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import signal

flag=True
def handler(signum, frame):
    global flag
    flag=False


x1=[]
y1=[]
action1=[]
t1=[]
theta1=[]
gamma1=[]

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
LOG_DIR = os.path.join(os.getcwd()+"/config/minitaur_trotting_env")
#print(LOG_DIR)
#16,17
CHECKPOINT = os.path.join(os.getcwd()+"/config/minituar_trotting_20200826/model.ckpt-17000000")

def main(argv):
  del argv  # Unused.
  fd=open("123.txt",mode='w',encoding='utf-8')
  signal.signal(signal.SIGINT,handler)
  config = utility.load_config(LOG_DIR)
  policy_layers = config.policy_layers
  value_layers = config.value_layers
  env = config.env(render=False)
  network = config.network
  imu=imu_BNO008X_uart.IMU("/dev/ttyTHS1")
  pos_contorl=position_control_threading.PositionControl()
  pos_contorl.Start()
  while pos_contorl.ready!=[1]*4 :
    pass
  t='a'


  with tf.Session() as sess:
    agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                             env,
                                             network,
                                             policy_layers=policy_layers,
                                             value_layers=value_layers,
                                             checkpoint=CHECKPOINT)

    sum_reward = 0


    start=time.time()
    while t!='t':
      t=input("please input t:")
    time.sleep(2)

    # observation = env.reset()
    # _last_frame_time = time.time()
    # while flag:
    #   action = agent.get_action([observation])
    #   action=[[0,0,0,0,0,0,0,0]]
    #   observation, reward, done, _, _action = env.step(action[0])
    #   pos_contorl.Run(_action)
    #   sum_reward += reward
    #   #time_spent = time.time() - _last_frame_time
    #   #print(imu.ReadDataMsg())
    #   #print(time_spent)
    #   #_last_frame_time = time.time()
    #   if time.time()-_last_frame_time >8:
    #     break

    observation = imu.ReadDataMsg()
    init_observation= observation
    _last_frame_time = time.time()
    while flag:
      observation=imu.ReadDataMsg()
      for i in range(4):
        fd.write(str(observation[i])+' ')
      fd.write(str(round(imu.x_acc,2))+' ')
      fd.write(str(round(imu.x_vel,2))+' ')
      fd.write(str(round(time.time()-_last_frame_time,2))+'\n')

      observation=[-observation[0]+init_observation[0],observation[1]-init_observation[1],observation[2],observation[3]]
      # observation=[0,0,0,0]
      action = agent.get_action([observation])
      _action,_ = env._transform_action_to_motor_command(action[0],time.time()-env._reset_time)
      pos_contorl.Run(_action)



      if time.time()-_last_frame_time >8:
        break

    pos_contorl.Stop()
    imu.device.close()



if __name__ == "__main__":
  tf.app.run(main)
