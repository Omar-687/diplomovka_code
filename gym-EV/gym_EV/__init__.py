from gym.envs.registration import register

register(
    id='EV-v0',
    entry_point='gym_EV.envs:EVEnv',
)

register(
    id='EV-v1',
    entry_point='gym_EV.envs:EVEnvOptim',
)

