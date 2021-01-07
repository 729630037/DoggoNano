import signal
import time

from drivers.imu_BNO008X_i2c import IMU      
from drivers.position_control import PositionControl
import tensorflow as tf
from tf_agents.trajectories import time_step as ts


USE_IMU=True
USE_REINFORCEMENT_LEARNING=False
DEBUG=False

imu=None

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

flag=True
def handler(signum, frame):
    global flag
    flag=False
signal.signal(signal.SIGINT,handler)


pos_control=PositionControl(signal_type="ik",mode="reactive")
pos_control.Start()
while pos_control.ready!=[1]*4 :
    pass
while input("please input t:")!='t':
    pass
pos_control.StopThread()
pos_control.odrives_thread.start()

if USE_IMU:
    imu=IMU()
    imu.imu_thread.start()

if USE_REINFORCEMENT_LEARNING:
    # saved_policy = tf.saved_model.load("policies/policy")
    # saved_policy = tf.saved_model.load("policies/reactive_policy")
    saved_policy = tf.saved_model.load("policies/trot_policy")    
    time_step=[0,0,0,[0,0,0,0]]

time_init=time.time()
stime=time_init

while flag:
    if USE_REINFORCEMENT_LEARNING:
        time_step=convert_to_tensor(time_step)
        action_step = saved_policy.action(time_step)
        proto_tensor=tf.make_tensor_proto(action_step.action)
        action=tf.make_ndarray(proto_tensor)
    else:
        action=[[0,0,0,0,0,0,0,0]]

    thetagamma=pos_control.TransformActionToMotorCommand(time.time()-time_init,action[0])  
    pos_control.odrive_queue.put(thetagamma)

    if USE_IMU:
        time_step=[0,0,0,imu.imu_queue.get()]
    else:
        time_step=[0,0,0,[0,0,0,0]]

    if DEBUG:
        print(time.time()-stime)
        stime=time.time()

    if time.time()-time_init>6:
        break

pos_control.Stop()


