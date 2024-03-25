from gym.envs.registration import register
# from envs.EV_env import EVEnv
# from envs.EV_env import
import gym
register(
    id='EV-v0',
    entry_point='gym_EV.envs:EVEnv',
)


# env = EVEnv()
# a = 0