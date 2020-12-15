# from envs.minitaur_reactive_env import MinitaurReactiveEnv
# from envs.minitaur_trotting_env import MinitaurTrottingEnv
# from envs.minitaur_gym_env import MinitaurGymEnv
#print("as")
#from . import minitaur
#from . import motor
#from . import minitaur_gym_env
import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

register(
    id='MinitaurTrottingEnv-v1',
    entry_point='envs.minitaur_trotting_env:MinitaurTrottingEnv',
    max_episode_steps=1000,
    reward_threshold=10.0,
)

register(
    id='MinitaurReactiveEnv-v1',
    entry_point='envs.minitaur_reactive_env:MinitaurReactiveEnv',
    max_episode_steps=1000,
    reward_threshold=10.0,
)    