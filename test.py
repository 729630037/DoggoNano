import matplotlib.pyplot as plt
import os,sys,inspect
import reverb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
tempdir = currentdir


import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import common
from tf_agents.utils import test_utils

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

env_name = "MinitaurTrottingEnv-v1"
eval_env = suite_pybullet.load(env_name,max_episode_steps=10000)
time_step= eval_env.reset()
saved_policy = tf.saved_model.load("policies/policy")
time_step=convert_to_tensor(time_step)
reward=0

# print(saved_policy.action(time_step))

# action = [0,0,0,0,0,0,0,0]
while not time_step.is_last():
    action_step = saved_policy.action(time_step)
    proto_tensor=tf.make_tensor_proto(action_step.action)
    action=tf.make_ndarray(proto_tensor)
    action=[[0,0,0,0,0,0,0,0]]
    time_step = eval_env.step(action[0])
    reward+=time_step[1]
    time_step=convert_to_tensor(time_step)

print("-----------------------")
print("reward: ", reward)
print("-----------------------")
# saved_policy = tf.saved_model.load('policy_4999')
# policy_state = saved_policy.get_initial_state(batch_size=1)

