"""Implements the gym environment of minitaur moving with trotting style.
"""
import os, inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)

if os.getcwd() not in sys.path:
  sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import math
import time
from gym import spaces
import numpy as np
from envs import minitaur_gym_env
from envs.env_randomizers.minitaur_env_randomizer_from_config import MinitaurEnvRandomizerFromConfig
from envs.env_randomizers.minitaur_push_randomizer import MinitaurPushRandomizer

# TODO(tingnan): These constants should be moved to minitaur/minitaur_gym_env.
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS
atime=time.time()

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

  def __init__(self,
               urdf_version=None,
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
               init_extension=2.1,    #2.0rad=114.59度
               init_swing=0.0,
               random_init_pose=False,
               step_frequency=2.0,
               extension_amplitude=0.8,   #0.35rad=20.05度 0.3rad=17.19度
               swing_amplitude=0.4,
               terrain_type="random",
               terrain_id='random',               
               ):
    """Initialize the minitaur trotting gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION,
        RAINBOW_DASH_V0_URDF_VERSION] are allowable versions. If None,
          DEFAULT_URDF_VERSION is used. Refer to minitaur_gym_env for more
          details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See minitaur.py for more details.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hung midair so
        that its walking gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      accurate_motor_model_enabled: Uses the nonlinear DC motor model if set to
        True.
      use_signal_in_observation: Includes the reference motor angles in the
        observation vector.
      use_angle_in_observation: Includes the measured motor angles in the
        observation vector.
      hard_reset: Whether to reset the whole simulation environment or just
        reposition the robot.
      env_randomizer: A list of EnvRandomizers that can randomize the
        environment during when env.reset() is called and add perturbation
        forces when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        minitaur_logging.proto.
      init_extension: The initial reset length of the leg.
      init_swing: The initial reset swing position of the leg.
      step_frequency: The desired leg stepping frequency.
      extension_amplitude: The maximum leg extension change within a locomotion
        cycle.
      swing_amplitude: The maximum leg swing change within a cycle.
    """

    # _swing_offset and _extension_offset is to mimick the bent legs. The
    # offsets will be added when applying the motor commands.
    self._swing_offset = np.zeros(NUM_LEGS)
    self._extension_offset = np.zeros(NUM_LEGS)
    self._random_init_pose=random_init_pose
    # The reset position.
    self._init_pose = [
        init_swing, init_swing, init_swing, init_swing, init_extension, init_extension,
        init_extension, init_extension
    ]
    self._flightPercent=0.5
    self._step_frequency = step_frequency
    self._extension_amplitude = extension_amplitude
    self._swing_amplitude = swing_amplitude
    self._use_signal_in_observation = use_signal_in_observation
    self._use_angle_in_observation = use_angle_in_observation
    super(MinitaurTrottingEnv,
          self).__init__(urdf_version=urdf_version,
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
                         terrain_type=terrain_type                       
                         )

    action_dim = NUM_LEGS * 2
    action_high = np.array([0.25] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    # For render purpose.
    self._cam_dist = 1.0
    self._cam_yaw = 30
    self._cam_pitch = -30
    self._motor_pose=np.zeros(NUM_MOTORS)

  def reset(self):
    # In this environment, the actions are
    # [swing leg 1, swing leg 2, swing leg 3, swing leg 4,
    #  extension leg 1, extension leg 2, extension leg 3, extension leg 4]
    if self._random_init_pose==True:
      self._init_pose = np.random.uniform(np.array(self._init_pose) - np.array([0.08]*8) ,
                                          np.array(self._init_pose) + np.array([0.08]*8)) 
    initial_motor_angles = self._convert_from_leg_model(self._init_pose)
    super(MinitaurTrottingEnv, self).reset(initial_motor_angles=initial_motor_angles,
                                           reset_duration=0.5)
    return self._get_observation()

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

  def _signal(self, t):
    """Generates the trotting gait for the robot.

    Args:
      t: Current time in simulation.

    Returns:
      A numpy array of the reference leg positions.

      sw_first_pair,ext_first_pair    0   2   sw_second_pair,ext_second_pair

      sw_second_pair,ext_second_pair  1   3   sw_first_pair,ext_first_pair

    """
    # Generates the leg trajectories for the two digonal pair of legs.
    ext_first_pair, sw_first_pair = self._gen_signal(t, 0)
    ext_second_pair, sw_second_pair = self._gen_signal(t, 0.5)

    trotting_signal = np.array([
        sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
        ext_second_pair, ext_second_pair, ext_first_pair
    ]) 
    signal = np.array(self._init_pose) + trotting_signal
    return signal

  def _transform_action_to_motor_command(self, action,time):
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
    # Add swing_offset and extension_offset to mimick the bent legs.
    action[0:NUM_LEGS] += self._swing_offset
    action[NUM_LEGS:2 * NUM_LEGS] += self._extension_offset

    # Add the reference trajectory (i.e. the trotting signal).
    #action += self._signal(self.minitaur.GetTimeSinceReset())
    action += self._signal(time)
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
