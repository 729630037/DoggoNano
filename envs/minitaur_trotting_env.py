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
from envs.kinematics import Kinematics
from envs.env_randomizers.minitaur_env_randomizer_from_config import MinitaurEnvRandomizerFromConfig
from envs.env_randomizers.minitaur_push_randomizer import MinitaurPushRandomizer

# TODO(tingnan): These constants should be moved to minitaur/minitaur_gym_env.
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS

class MinitaurTrottingEnv(minitaur_gym_env.MinitaurGymEnv):
  """The trotting gym environment for the minitaur.

  In this env, Minitaur performs a trotting style locomotion specified by
  extension_amplitude, swing_amplitude, and step_frequency. Each diagonal pair
  of legs will move according to the reference trajectory:
      extension = extsion_amplitude * cos(2 * pi * step_frequency * t + phi)
      swing = swing_amplitude * sin(2 * pi * step_frequency * t + phi)
  And the two diagonal leg pairs have a phase (phi) difference of pi. The
  reference signal may be modified by the feedback actiones from a balance
  controller (e.g. a neural network).

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 166}
  load_ui = True
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
               signal_type="ik",
               random_init_pose=False,
               step_frequency=2.0,
               init_theta=0.0,
               init_gamma=1.4,
               theta_amplitude=0.8,   #0.35rad=20.05度 0.3rad=17.19度
               gamma_amplitude=0.4,
               terrain_type="random",
               terrain_id='random',
               mark='base'                              
               ):
    """Initialize the minitaur trotting gym environment."""

    # _swing_offset and _extension_offset is to mimick the bent legs. The
    # offsets will be added when applying the motor commands.
    self._swing_offset = np.zeros(NUM_LEGS)
    self._extension_offset = np.zeros(NUM_LEGS)
    self._random_init_pose=random_init_pose
    # The reset position.
    self._init_pose = [
        init_theta, init_theta, init_theta, init_theta, init_gamma, init_gamma,
        init_gamma, init_gamma
    ]
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
                         debug=debug,
                         mark=mark
                         )

    # (eventually) allow different feedback ranges/action spaces for different signals
    action_max = {
        'ik': 0.4,
        'ol': 0.01
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
    self._cam_yaw = 0.0
    self._cam_pitch = -20

    self._gait_planner = GaitPlanner("trot")
    self._kinematics = Kinematics()
    self.goal_reached = False
    self._stay_still = False
    self.is_terminating = False

  def reset(self):
    if self._random_init_pose==True:
      self._init_pose = np.random.uniform(np.array(self._init_pose) - np.array([0.08]*8) ,
                                          np.array(self._init_pose) + np.array([0.08]*8)) 
    initial_motor_angles = self._convert_from_leg_model(self._init_pose)
    super(MinitaurTrottingEnv, self).reset(initial_motor_angles=initial_motor_angles,
                                           reset_duration=0.5)
    self.goal_reached = False
    self.is_terminating = False
    self._stay_still = False
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
    if not self._target_position or self._random_pos_target:
        bound = -3 if self.backwards else 3
        self._target_position = random.uniform(bound//2, bound)
        self._random_pos_target = True
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

  def _check_target_position(self, t):
      if self._target_position:
          current_x = abs(self.minitaur.GetBasePosition()[0])
          # give 0.15 stop space
          if current_x >= abs(self._target_position) - 0.15:
              self.goal_reached = True
              if not self.is_terminating:
                  self.end_time = t
                  self.is_terminating = True

  @staticmethod
  def _evaluate_base_stage_coeff(current_t, end_t=0.0, width=0.001):
      # sigmoid function
      beta = p = width
      if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
          return (2 / beta ** 2) * (current_t - p + beta) ** 2
      elif p - (beta/2) + end_t <= current_t <= p + end_t:
          return 1 - (2 / beta ** 2) * (current_t - p) ** 2
      else:
          return 1

  @staticmethod
  def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
      # ramp function(斜坡函数)
      p = 0.8 + action[0]
      if end_t <= current_t <= p + end_t:
          return current_t
      else:
          return 1.0

  @staticmethod
  def _evaluate_brakes_stage_coeff(current_t, action, end_t=0.0, end_value=0.0):
      # ramp function
      p = 0.8 + action[1]
      if end_t <= current_t <= p + end_t:
          return 1 - (current_t - end_t)
      else:
          return end_value

  def _signal(self, t, action):
      if self._signal_type == 'ik':
          return self._IK_signal(t, action)
      if self._signal_type == 'ol':
          return self._open_loop_signal(t, action)

  def _IK_signal(self, t, action):
      base_pos_coeff = self._evaluate_base_stage_coeff(t, width=1.5)
      gait_stage_coeff = self._evaluate_gait_stage_coeff(t, action)
      step = 0.6
      period = 0.65
      base_x = self._base_x
      if self.backwards:
          step = -.3
          period = .5
          base_x = .0
      if self._is_render and self._is_debug:
          position, orientation, step_length, step_rotation, step_angle, step_period = \
              self._read_inputs(base_pos_coeff, gait_stage_coeff)
      else:
          position = np.array([base_x,
                                self._base_y * base_pos_coeff,
                                self._base_z * base_pos_coeff])
          orientation = np.array([self._base_roll * base_pos_coeff,
                                  self._base_pitch * base_pos_coeff,
                                  self._base_yaw * base_pos_coeff])
          step_length = (self.step_length if self.step_length is not None else step) * gait_stage_coeff
          step_rotation = (self.step_rotation if self.step_rotation is not None else 0.0)
          step_angle = self.step_angle if self.step_angle is not None else 0.0
          step_period = (self.step_period if self.step_period is not None else period)
      if self.goal_reached:
          brakes_coeff = self._evaluate_brakes_stage_coeff(t, action, self.end_time)
          step_length *= brakes_coeff
          if brakes_coeff == 0.0:
              self._stay_still = True
      direction = -1.0 if step_length < 0 else 1.0
      frames = self._gait_planner.loop(step_length, step_angle, step_rotation, step_period, direction)
      fr_angles, fl_angles, rr_angles, rl_angles, _ = self._kinematics.solve(orientation, position, frames)
      signal = [
          fl_angles[0], fl_angles[1], fl_angles[2],
          fr_angles[0], fr_angles[1], fr_angles[2],
          rl_angles[0], rl_angles[1], rl_angles[2],
          rr_angles[0], rr_angles[1], rr_angles[2]
      ]
      return signal

  def _open_loop_signal(self, t, action):

    # Generates the leg trajectories for the two digonal pair of legs.
    ext_first_pair, sw_first_pair = self._gen_signal(t, 0)
    ext_second_pair, sw_second_pair = self._gen_signal(t, 0.5)

    trotting_signal = np.array([
        sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
        ext_second_pair, ext_second_pair, ext_first_pair
    ]) 
    signal = np.array(self._init_pose) + trotting_signal
    return signal

  def _convert_from_leg_model(self, leg_pose):
    """Converts leg space action into motor commands.

    Args:
      leg_pose: A numpy array. leg_pose[0:NUM_LEGS] are leg swing angles
        and leg_pose[NUM_LEGS:2*NUM_LEGS] contains leg extensions.

    Returns:
      A numpy array of the corresponding motor angles for the given leg pose.
        θ1=e-s   θ2=e+s

      action[0]=ext_first_pair-sw_first_pair    0   2   action[4]=ext_second_pair+sw_second_pair
      action[1]=ext_first_pair+sw_first_pair            action[5]=ext_second_pair-sw_second_pair
      
      action[2]=ext_second_pair-sw_second_pair  1   3   action[6]=ext_first_pair+sw_first_pair    
      action[3]=ext_second_pair+sw_second_pair          action[7]=ext_first_pair-sw_first_pair

    """
    motor_pose = np.zeros(NUM_MOTORS)
    for i in range(NUM_LEGS):
      motor_pose[int(2 * i)] = leg_pose[NUM_LEGS + i] - (-1)**int(i / 2) * leg_pose[i]
      motor_pose[int(2 * i + 1)] = leg_pose[NUM_LEGS + i] + (-1)**int(i / 2) * leg_pose[i]
    return motor_pose

  def _gen_signal(self, t, phase):
    """Generates a sinusoidal reference leg trajectory.

    The foot (leg tip) will move in a ellipse specified by extension and swing
    amplitude.

    Args:
      t: Current time in simulation.
      phase: The phase offset for the periodic trajectory.

    Returns:
      The desired leg extension and swing angle at the current time.
    """
    gp=(t*self._step_frequency+phase)%1
    if gp<= self._flightPercent:
        extension = -self._extension_amplitude * math.sin(math.pi/self._flightPercent* gp)
        swing = self._swing_amplitude* math.cos(math.pi/self._flightPercent* gp) 
    else:
        percentBack = (gp-self._flightPercent)/(1.0-self._flightPercent)
        extension = (1-self._extension_amplitude)* math.sin(math.pi*percentBack)
        swing = self._swing_amplitude * math.cos(math.pi*percentBack+math.pi) 
    return extension, swing

  def _signal(self, t, action):
    """Generates the trotting gait for the robot.

    Args:
      t: Current time in simulation.

    Returns:
      A numpy array of the reference leg positions.

      sw_first_pair,ext_first_pair    0   2   sw_second_pair,ext_second_pair

      sw_second_pair,ext_second_pair  1   3   sw_first_pair,ext_first_pair

    """
    if self._signal_type == 'ik':
        return self._IK_signal(t, action)
    if self._signal_type == 'ol':
        return self._open_loop_signal(t, action)

  def _transform_action_to_motor_command(self, action,t):
    """Generates the motor commands for the given action.

    Swing/extension offsets and the reference leg trajectory will be added on
    top of the inputs before the conversion.

    Args:
      action: A numpy array contains the leg swings and extensions that will be
        added to the reference trotting trajectory. action[0:NUM_LEGS] are leg
        swing angles, and action[NUM_LEGS:2*NUM_LEGS] contains leg extensions.

    Returns:
      A numpy array of the desired motor angles for the given leg space action.
    """
    if self._stay_still:
        return self.init_pose    
    # Add swing_offset and extension_offset to mimick the bent legs.
    action[0:NUM_LEGS] += self._swing_offset
    action[NUM_LEGS:2 * NUM_LEGS] += self._extension_offset

    # Add the reference trajectory (i.e. the trotting signal).
    #action += self._signal(self.minitaur.GetTimeSinceReset())
    self._check_target_position(t)    
    action += self._signal(t,action)
    for i in range(0,4):
      np.clip(action[i],-0.45,0.45)
    for i in range(4,8):
      np.clip(action[i],0.85,2.35)    
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
      _,action=self._transform_action_to_motor_command([0] * 8,time.time()-self._reset_time)      
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
      _,action=self._transform_action_to_motor_command([0] * 8,time.time()-self._reset_time)      
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

  def set_swing_offset(self, value):
    """Set the swing offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
    self._swing_offset = value

  def set_extension_offset(self, value):
    """Set the extension offset of each leg.

    It is to mimic the bent leg.

    Args:
      value: A list of four values.
    """
    self._extension_offset = value
