from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
tmp_path = "/tmp/sac/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = SAC("MlpPolicy", "Pendulum-v1",tensorboard_log="/tmp/sac/", verbose=1)
# Set new logger
model.set_logger(new_logger)
model.learn(10000)