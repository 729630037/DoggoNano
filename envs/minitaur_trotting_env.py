"""Implements the gym environment of minitaur moving with trotting style.
"""
import os, inspect,sys
import math
import time
import random

from gym import spaces
import numpy as np
from envs import minitaur_gym_env
from envs.gait_planner import GaitPlanner 
from envs import kinematics
from envs.env_randomizers.minitaur_env_randomizer_from_config import MinitaurEnvRandomizerFromConfig
from envs.env_randomizers.minitaur_push_randomizer import MinitaurPushRandomizer
from drivers.position_control import PositionControl

# TODO(tingnan): These constants should be moved to minitaur/minitaur_gym_env.
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS

class MinitaurTrottingEnv(minitaur_gym_env.MinitaurGymEnv):
  """The trotting gym environment for the minitaur.

  In this env, Minitaur performs a trotting style locomotion specified by
  gamma_amplitude, theta_amplitude, and step_frequency. Each diagonal pair
  of legs will move according to the reference trajectory:
      gamma = gammasion_amplitude * cos(2 * pi * step_frequency * t + phi)
      theta = theta_amplitude * sin(2 * pi * step_frequency * t + phi)
  And the two diagonal leg pairs have a phase (phi) difference of pi. The
  reference signal may be modified by the feedback actiones from a balance
  controller (e.g. a neural network).

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 166}
  load_ui = False
  is_terminating = False
  def __init__(self,
              debug=False,
              control_time_step=0.001,
              action_repeat=1,
              control_latency=0.03,
              pd_latency=0.003,
              on_rack=False,
              motor_kp=1.0,
              motor_kd=0.015,
              remove_default_joint_damping=True,
              render=True,
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

    # The reset position.
    self._init_pose = [
        init_theta, init_theta, init_theta, init_theta, init_gamma, init_gamma,
        init_gamma, init_gamma
    ]
    """Initialize the minitaur trotting gym environment."""
    # _theta_offset and _gamma_offset is to mimick the bent legs. The
    # offsets will be added when applying the motor commands.
    self._theta_offset = np.zeros(NUM_LEGS)
    self._gamma_offset = np.zeros(NUM_LEGS)
    self._random_init_pose=random_init_pose

    self._flightPercent=0.5
    self._step_frequency = step_frequency
    self._theta_amplitude = theta_amplitude
    self._gamma_amplitude = gamma_amplitude
    self._use_signal_in_observation = use_signal_in_observation
    self._use_angle_in_observation = use_angle_in_observation
    self._signal_type = signal_type

    super(MinitaurTrottingEnv,
          self).__init__(
                        accurate_motor_model_enabled=accurate_motor_model_enabled,
                        motor_overheat_protection=False,
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
                        action_repeat=action_repeat,
                        terrain_id=terrain_id,
                        terrain_type=terrain_type,
                        target_position=target_position,
                        signal_type=signal_type,
                        backwards=backwards,
                        debug=debug
                        )


    # (eventually) allow different feedback ranges/action spaces for different signals
    action_max = {
        'ik': 0.4,
        'ol': 0.15
    }
    action_dim_map = {
        'ik': 2,
        'ol': 8
    }
    action_dim = action_dim_map[self._signal_type]
    action_high = np.array([action_max[self._signal_type]] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    # For render purpose.
    self._cam_dist = 1.0
    self._cam_yaw = 3.0
    self._cam_pitch = -30

    self._gait_planner = GaitPlanner("trot")
    self._kinematics = kinematics.Kinematics()
    self._position_control=PositionControl(signal_type=self._signal_type)
    self.goal_reached = False
    self._stay_still = stay_still
    self.is_terminating = False
    self._fd=open("dd.txt","w")


  def reset(self):

    initial_motor_angles = self._convert_from_leg_model(self._init_pose)
    super(MinitaurTrottingEnv, self).reset(initial_motor_angles=initial_motor_angles,
                                          reset_duration=0.5)
    if self._random_init_pose==True:
      self._init_pose = np.random.uniform(np.array(self._init_pose) - np.array([0.08]*8) ,
                                          np.array(self._init_pose) + np.array([0.08]*8)) 
    self.goal_reached = False
    self.is_terminating = False
    # self._stay_still = False
    if self._backwards is None:
      self.backwards = random.choice([True, False])
    else:
      self.backwards = self._backwards
    step = 0.6
    period = 0.65
    base_x = self._base_x
    if self.backwards:
        step = -.3
        period = .5
        base_x = .0
    # if not self._target_position or self._random_pos_target:
    #     bound = -3 if self.backwards else 3
    #     self._target_position = random.uniform(bound//2, bound)
    #     self._random_pos_target = True
    if self._is_render and self._signal_type == 'ik':
        if self.load_ui:
            self.setup_ui(base_x, step, period)
            self.load_ui = False
    if self._is_debug:
        print(f"Target Position x={self._target_position}, Random assignment: {self._random_pos_target}, Backwards: {self.backwards}")        
    return self._get_observation()


  def setup_ui(self, base_x, step, period):
    self.base_x_ui = self._pybullet_client.addUserDebugParameter("base_x",
                                                                  self._ranges["base_x"][0],
                                                                  self._ranges["base_x"][1],
                                                                  base_x)
    self.base_y_ui = self._pybullet_client.addUserDebugParameter("base_y",
                                                                  self._ranges["base_y"][0],
                                                                  self._ranges["base_y"][1],
                                                                  self._ranges["base_y"][2])
    self.base_z_ui = self._pybullet_client.addUserDebugParameter("base_z",
                                                                  self._ranges["base_z"][0],
                                                                  self._ranges["base_z"][1],
                                                                  self._ranges["base_z"][2])
    self.roll_ui = self._pybullet_client.addUserDebugParameter("roll",
                                                                self._ranges["roll"][0],
                                                                self._ranges["roll"][1],
                                                                self._ranges["roll"][2])
    self.pitch_ui = self._pybullet_client.addUserDebugParameter("pitch",
                                                                self._ranges["pitch"][0],
                                                                self._ranges["pitch"][1],
                                                                self._ranges["pitch"][2])
    self.yaw_ui = self._pybullet_client.addUserDebugParameter("yaw",
                                                              self._ranges["yaw"][0],
                                                              self._ranges["yaw"][1],
                                                              self._ranges["yaw"][2])
    self.step_length_ui = self._pybullet_client.addUserDebugParameter("step_length", -0.7, 0.7, step)
    self.step_rotation_ui = self._pybullet_client.addUserDebugParameter("step_rotation", -1.5, 1.5, 0.)
    self.step_angle_ui = self._pybullet_client.addUserDebugParameter("step_angle", -180., 180., 0.)
    self.step_period_ui = self._pybullet_client.addUserDebugParameter("step_period", 0.2, 0.9, period)

  def _read_inputs(self, base_pos_coeff, gait_stage_coeff):
      position = np.array(
          [
              self._pybullet_client.readUserDebugParameter(self.base_x_ui),
              self._pybullet_client.readUserDebugParameter(self.base_y_ui) * base_pos_coeff,
              self._pybullet_client.readUserDebugParameter(self.base_z_ui) * base_pos_coeff
          ]
      )
      orientation = np.array(
          [
              self._pybullet_client.readUserDebugParameter(self.roll_ui) * base_pos_coeff,
              self._pybullet_client.readUserDebugParameter(self.pitch_ui) * base_pos_coeff,
              self._pybullet_client.readUserDebugParameter(self.yaw_ui) * base_pos_coeff
          ]
      )
      step_length = self._pybullet_client.readUserDebugParameter(self.step_length_ui) * gait_stage_coeff
      step_rotation = self._pybullet_client.readUserDebugParameter(self.step_rotation_ui)
      step_angle = self._pybullet_client.readUserDebugParameter(self.step_angle_ui)
      step_period = self._pybullet_client.readUserDebugParameter(self.step_period_ui)
      return position, orientation, step_length, step_rotation, step_angle, step_period


  def _convert_from_leg_model(self, leg_pose):
    """Converts leg space action into motor commands.

    Args:
      leg_pose: A numpy array. leg_pose[0:NUM_LEGS] are leg theta angles
        and leg_pose[NUM_LEGS:2*NUM_LEGS] contains leg gammas.

    Returns:
      A numpy array of the corresponding motor angles for the given leg pose.
        θ1=pi-e-s   θ2=pi-e+s

      action[0]=gamma_first-theta_first    0   2   action[4]=gamma_second+theta_second
      action[1]=gamma_first+theta_first            action[5]=gamma_second-theta_second
      
      action[2]=gamma_second-theta_second  1   3   action[6]=gamma_first+theta_first    
      action[3]=gamma_second+theta_second          action[7]=gamma_first-theta_first

    """
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
    # Add theta_offset and gamma_offset to mimick the bent legs.
    action[0:NUM_LEGS] += self._theta_offset
    action[NUM_LEGS:2 * NUM_LEGS] += self._gamma_offset
    t= time.time()-self._reset_time 
    action = self._position_control.Signal(t,action)  
    # x,y=self._kinematics.solve_K([action[0],action[4]])
    # self._fd.write(str(x)+" "+str(y)+'\n') 
    for i in range(0,4):
      action[i]=np.clip(action[i],-0.6,0.6)
    for i in range(4,8):
      action[i]=np.clip(action[i],0.45,2.45)    
    return action,self._convert_from_leg_model(action)

  def is_fallen(self):
    """Decide whether the minitaur has fallen.

    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
    roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
    is_fallen = math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.3
    return is_fallen

  def _get_true_observation(self):
    """Get the true observations of this environment.

    It includes the true roll, pitch, roll dot and pitch dot of the base. Also
    includes the disired/observed motor angles if the relevant flags are set.

    Returns:
      The observation list.
    """
    observation = []
    roll, pitch, _ = self.minitaur.GetTrueBaseRollPitchYaw()
    roll_rate, pitch_rate, _ = self.minitaur.GetTrueBaseRollPitchYawRate()
    observation.extend([roll, pitch, roll_rate, pitch_rate])
    if self._use_signal_in_observation:
      _,action=self._transform_action_to_motor_command([0] * 8)      
      observation.extend(action)
    if self._use_angle_in_observation:
      observation.extend(self.minitaur.GetMotorAngles().tolist())
    self._true_observation = np.array(observation)
    return self._true_observation

  def _get_observation(self):
    """Get observations of this environment.

    It includes the base roll, pitch, roll dot and pitch dot which may contain
    noises, bias, and latency. Also includes the disired/observed motor angles
    if the relevant flags are set.

    Returns:
      The observation list.
    """
    observation = []
    roll, pitch, _ = self.minitaur.GetBaseRollPitchYaw()
    roll_rate, pitch_rate, _ = self.minitaur.GetBaseRollPitchYawRate()
    observation.extend([roll, pitch, roll_rate, pitch_rate])
    if self._use_signal_in_observation:
      _,action=self._transform_action_to_motor_command([0] * 8)      
      observation.extend(action)
    if self._use_angle_in_observation:
      observation.extend(self.minitaur.GetMotorAngles().tolist())
    self._observation = np.array(observation)
    return self._observation

  def _get_observation_upper_bound(self):
    """Get the upper bound of the observation.

    Returns:
      A numpy array contains the upper bound of an observation. See
      GetObservation() for the details of each element of an observation.
    """
    upper_bound = []
    upper_bound.extend([2 * math.pi] * 2)  # Roll, pitch, yaw of the base.
    upper_bound.extend([2 * math.pi / self._time_step/1000] * 2)  # Roll, pitch, yaw rate.
    if self._use_signal_in_observation:
      upper_bound.extend([2 * math.pi] * NUM_MOTORS)  # Signal
    if self._use_angle_in_observation:
      upper_bound.extend([2 * math.pi] * NUM_MOTORS)  # Motor angles
    return np.array(upper_bound)

  def _get_observation_lower_bound(self):
    """Get the lower bound of the observation.

    Returns:
      The lower bound of an observation (the reverse of the upper bound).
    """
    lower_bound = -self._get_observation_upper_bound()
    return lower_bound

  def set_theta_offset(self, value):
    """Set the theta offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
    self._theta_offset = value

  def set_gamma_offset(self, value):
    """Set the gamma offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
    self._gamma_offset = value
