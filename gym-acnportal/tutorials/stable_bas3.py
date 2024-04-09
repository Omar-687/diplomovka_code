import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.monitor import Monitor


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

# the evaluation works with only saved model so save is only necessary


# it prints live with this tensorboard

env = gym.make("Pendulum-v1", render_mode
="human")
env = Monitor(env)
model = SAC("MlpPolicy", env,tensorboard_log="/tmp/sac/", verbose=2)
# Random Agent, before training
print('aa')
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30, warn=False)




print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



# model.learn(total_timesteps=10000,callback=FigureRecorderCallback())
# model.save("sac_pendulum")

del model # remove to demonstrate saving and loading



# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model = SAC.load("sac_pendulum")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()