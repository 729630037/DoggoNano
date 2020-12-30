"""This file implements the gym environment of minitaur alternating legs.

"""

import os, inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)

if os.getcwd() not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(os.getcwd())))


from envs.gait_planner import GaitPlanner 
from envs import kinematics
import collections
import math
import time
from gym import spaces
import numpy as np
from envs import minitaur_gym_env
from envs.env_randomizers.minitaur_env_randomizer_from_config import MinitaurEnvRandomizerFromConfig
from envs.env_randomizers.minitaur_push_randomizer import MinitaurPushRandomizer
from drivers.position_control import PositionControl

INIT_EXTENSION_POS = 2.0
INIT_SWING_POS = 0.0
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS

MinitaurPose = collections.namedtuple(
    "MinitaurPose", "swing_angle_1, swing_angle_2, swing_angle_3, swing_angle_4, "
    "extension_angle_1, extension_angle_2, extension_angle_3, "
    "extension_angle_4")


class MinitaurReactiveEnv(minitaur_gym_env.MinitaurGymEnv):
  """The gym environment for the minitaur.

  It simulates the locomotion of a minitaur, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 166}

  def __init__( self,
              debug=False,
              urdf_version=None,
              energy_weight=0.005,              
              control_time_step=0.001,
              action_repeat=1,
              control_latency=0.03,
              pd_latency=0.003,
              on_rack=False,
              motor_kp=1.0,
              motor_kd=0.015,
              remove_default_joint_damping=True,
              render=False,
              num_steps_to_log=1000,
              accurate_motor_model_enabled=True,
              use_signal_in_observation=False,
              use_angle_in_observation=False,
              hard_reset=False,
              env_randomizer=[MinitaurEnvRandomizerFromConfig,MinitaurPushRandomizer],
              log_path=None,
              target_position=None,
              backwards=None,
              signal_type="ol",
              random_init_pose=False,
              stay_still=False,
              step_frequency=2.0,
              init_theta=0.0,
              theta_amplitude=0.4,   #0.35rad=20.05度 0.3rad=17.19度
              init_gamma=1.1,
              gamma_amplitude=0.8,
              terrain_type="plane",
              terrain_id='random'
              ):
    """Initialize the minitaur trotting gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        minitaur_gym_env for more details.
      energy_weight: The weight of the energy term in the reward function. Refer
        to minitaur_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See minitaur.py for more details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur"s base is hung midair so
        that its walking gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      accurate_motor_model_enabled: Whether to use the accurate motor model from
        system identification. Refer to minitaur_gym_env for more details.
      use_angle_in_observation: Whether to include motor angles in observation.
      hard_reset: Whether hard reset (swipe out everything and reload) the
        simulation. If it is false, the minitaur is set to the default pose
        and moved to the origin.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        minitaur_logging.proto.
    """   
    super(MinitaurReactiveEnv,
          self).__init__(urdf_version=urdf_version,
                         energy_weight=energy_weight,
                         accurate_motor_model_enabled=accurate_motor_model_enabled,
                         motor_overheat_protection=True,
                         motor_kp=motor_kp,
                         motor_kd=motor_kd,
                         remove_default_joint_damping=remove_default_joint_damping,
                         control_latency=control_latency,
                         pd_latency=pd_latency,
                         on_rack=on_rack,
                         render=render,
                         hard_reset=hard_reset,
                         num_steps_to_log=num_steps_to_log,
                         env_randomizer=env_randomizer,
                         log_path=log_path,
                         control_time_step=control_time_step,
                         action_repeat=action_repeat)

    # (eventually) allow different feedback ranges/action spaces for different signals
    action_max = {
        'ik': 0.4,
        'ol': 0.3
    }
    action_dim_map = {
        'ik': 2,
        'ol': 4,
    }
    action_dim = action_dim_map[self._signal_type]
    action_low = np.array([action_max[self._signal_type]] * action_dim)
    action_high = -action_low
    self.action_space = spaces.Box(action_low, action_high)

    self._flightPercent=0.5
    self._step_frequency = step_frequency
    self._theta_amplitude = theta_amplitude
    self._gamma_amplitude = gamma_amplitude
    self._use_signal_in_observation = use_signal_in_observation
    self._use_angle_in_observation = use_angle_in_observation
    self._signal_type = signal_type
    self._use_angle_in_observation = use_angle_in_observation
    self._init_pose = [
        init_theta, init_theta, init_theta, init_theta, init_gamma, init_gamma,
        init_gamma, init_gamma
    ] 
    self._signal_type = signal_type
    self._gait_planner = GaitPlanner("gallop")
    self._kinematics = kinematics.Kinematics()
    self._cam_dist = 1.0
    self._cam_yaw = 30
    self._cam_pitch = -30

  def reset(self):
    initial_motor_angles = self._convert_from_leg_model(self._init_pose)
    super(MinitaurReactiveEnv, self).reset(initial_motor_angles=initial_motor_angles,
                                           reset_duration=0.5)
    return self._get_observation()

  @staticmethod
  def _evaluate_stage_coefficient(current_t, end_t=0.0, width=0.001):
      # sigmoid function
      beta = p = width
      if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
          return (2 / beta ** 2) * (current_t - p + beta) ** 2
      elif p - (beta/2) + end_t <= current_t <= p + end_t:
          return 1 - (2 / beta ** 2) * (current_t - p) ** 2
      else:
          return 1

  @staticmethod
  def _evaluate_brakes_stage_coeff(current_t, action, end_t=0.0, end_value=0.0):
      # ramp function
      p = 1. + action[0]
      if end_t <= current_t <= p + end_t:
          return 1 - (current_t - end_t)
      else:
          return end_value

  @staticmethod
  def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
      # ramp function
      p = 1. + action[1]
      if end_t <= current_t <= p + end_t:
          return current_t
      else:
          return 1.0

  def _signal(self, t, action):
      if self._signal_type == 'ik':
          return self._IK_signal(t, action)
      if self._signal_type == 'ol':
          return self._open_loop_signal(t, action)

  def _IK_signal(self, t, action):
      gait_stage_coeff = self._evaluate_gait_stage_coeff(t, action)
      position = np.array([0,0,0])
      orientation = np.array([0,0,0])
      step_length = self.step_length * gait_stage_coeff
      step_rotation = (self.step_rotation if self.step_rotation is not None else 0.0)
      step_angle = self.step_angle if self.step_angle is not None else 0.0
      step_period = self.step_period

      frames = self._gait_planner.loop(step_length, step_angle, step_rotation, step_period, 1.0)
      fr_angles, fl_angles, br_angles, bl_angles, _ = self._kinematics.solve(orientation, position, frames)
      signal = np.array([fl_angles[0],bl_angles[0],fr_angles[0],br_angles[0],
                  fl_angles[1],bl_angles[1],fr_angles[1],br_angles[1]])
      return signal

  def _open_loop_signal(self, t, action):
      signal = np.array(self._init_pose)+action
      return signal

  def _convert_from_leg_model(self, leg_pose):
    motor_pose = np.zeros(NUM_MOTORS)
    for i in range(NUM_LEGS):
      motor_pose[int(2 * i)] = math.pi-leg_pose[NUM_LEGS + i] - (-1)**int(i / 2) * leg_pose[i]
      motor_pose[int(2 * i + 1)] = math.pi-leg_pose[NUM_LEGS + i] + (-1)**int(i / 2) * leg_pose[i]
    return motor_pose

  def _transform_action_to_motor_command(self, action):
    """
    Generates the motor commands for the given action.
    theta/gamma offsets and the reference leg trajectory will be added on
    top of the inputs before the conversion.
    """
    if self._stay_still:
        return self._init_pose,self._convert_from_leg_model(self._init_pose)    

    t= time.time()-self._reset_time 
    action = self._signal(t,action)  
    # x,y=self._kinematics.solve_K([action[0],action[4]])
    # self._fd.write(str(x)+" "+str(y)+'\n') 
    for i in range(0,4):
      action[i]=np.clip(action[i],-0.6,0.6)
    for i in range(4,8):
      action[i]=np.clip(action[i],0.45,2.45)    
    return action,self._convert_from_leg_model(action)

  def is_fallen(self):
    """Decides whether the minitaur is in a fallen state.

    If the roll or the pitch of the base is greater than 0.3 radians, the
    minitaur is considered fallen.

    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
    roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
    return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.3

  def _get_true_observation(self):
    """Get the true observations of this environment.

    It includes the roll, the pitch, the roll dot and the pitch dot of the base.
    If _use_angle_in_observation is true, eight motor angles are added into the
    observation.

    Returns:
      The observation list, which is a numpy array of floating-point values.
    """
    roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
    roll_rate, pitch_rate, _ = self.minitaur.GetTrueBaseRollPitchYawRate()
    observation = [roll, pitch, roll_rate, pitch_rate]
    if self._use_angle_in_observation:
      observation.extend(self.minitaur.GetMotorAngles().tolist())
    self._true_observation = np.array(observation)
    return self._true_observation

  def _get_observation(self):
    roll, pitch, _ = self.minitaur.GetBaseRollPitchYaw()
    roll_rate, pitch_rate, _ = self.minitaur.GetBaseRollPitchYawRate()
    observation = [roll, pitch, roll_rate, pitch_rate]
    if self._use_angle_in_observation:
      observation.extend(self.minitaur.GetMotorAngles().tolist())
    self._observation = np.array(observation)
    return self._observation

  def _get_observation_upper_bound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See _get_true_observation() for the
      details of each element of an observation.
    """
    upper_bound_roll = 2 * math.pi
    upper_bound_pitch = 2 * math.pi
    upper_bound_roll_dot = 2 * math.pi / self._time_step
    upper_bound_pitch_dot = 2 * math.pi / self._time_step
    upper_bound_motor_angle = 2 * math.pi
    upper_bound = [
        upper_bound_roll, upper_bound_pitch, upper_bound_roll_dot, upper_bound_pitch_dot
    ]

    if self._use_angle_in_observation:
      upper_bound.extend([upper_bound_motor_angle] * NUM_MOTORS)
    return np.array(upper_bound)

  def _get_observation_lower_bound(self):
    lower_bound = -self._get_observation_upper_bound()
    return lower_bound
