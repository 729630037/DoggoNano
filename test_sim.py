import os,sys,inspect
import envs.tools.bullet_client
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import suite_pybullet

USE_REINFORCEMENT_LEARNING=True

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
eval_env = suite_pybullet.load(env_name,max_episode_steps=3000)
time_step= eval_env.reset()

if USE_REINFORCEMENT_LEARNING:
    saved_policy = tf.saved_model.load("policies/policy")
    time_step=convert_to_tensor(time_step)
reward=0

# print(saved_policy.action(time_step))

# action = [0,0,0,0,0,0,0,0]
while not time_step.is_last():
    if USE_REINFORCEMENT_LEARNING: 
        time_step=convert_to_tensor(time_step)
        action_step = saved_policy.action(time_step)
        proto_tensor=tf.make_tensor_proto(action_step.action)
        action=tf.make_ndarray(proto_tensor)
    else:
        action=[[0,0,0,0,0,0,0,0]]
    time_step = eval_env.step(action[0])
    reward+=time_step[1]

print("-----------------------")
print("reward: ", reward)
print("-----------------------")
# saved_policy = tf.saved_model.load('policy_4999')
# policy_state = saved_policy.get_initial_state(batch_size=1)

