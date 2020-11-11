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
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math

x1=[]
y1=[]
action1=[]
t1=[]
theta1=[]
gamma1=[]

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
LOG_DIR = os.path.join(os.getcwd()+"/config/minitaur_trotting_env")
print(LOG_DIR)
CHECKPOINT = os.path.join(os.getcwd()+"/config/minituar_trotting_20200823/model.ckpt-16000000")

def main(argv):
  del argv  # Unused.
  config = utility.load_config(LOG_DIR)
  policy_layers = config.policy_layers
  value_layers = config.value_layers
  env = config.env(render=True)
  network = config.network
  with tf.Session() as sess:
    agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                             env,
                                             network,
                                             policy_layers=policy_layers,
                                             value_layers=value_layers,
                                             checkpoint=CHECKPOINT)

    sum_reward = 0
    observation = env.reset()
    _last_frame_time=time.time()
    last_frame_time=_last_frame_time
    # for i in range(2000):
    #   action = agent.get_action([observation])
    #   action=[[0,0,0,0,0,0,0,0]]
    #   observation, reward, done, _action = env.step(action[0])
    #   sum_reward += reward
    #   time_spent = time.time() - _last_frame_time
    #   #print(time_spent)
    #   _last_frame_time = time.time()         
    #   if done:
    #     break

    for i in range(2000):
      action = agent.get_action([observation])
      # action=[[0,0,0,0,0,0,0,0]]
      observation, reward, done,_action = env.step(action[0])
      # print(vel)
      sum_reward += reward         
      t1.append(time.time() - last_frame_time)
      gamma1.append(observation[0]) 
      # theta1.append(_action[0])
      # time1= imu.read_data_msg()
      # print(time.time() - _last_frame_time)
      # _last_frame_time=time.time()
      if done:
        break

    print(time.time() - last_frame_time)
    tf.logging.info("reward: %s", sum_reward)
    plt.figure()
    plt.plot(t1, gamma1) 
    # # plt.plot(t1, theta1)     
    plt.show()


if __name__ == "__main__":
  tf.app.run(main)
