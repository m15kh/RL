from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym



env = Monitor(gym.make("LunarLander-v3"))

model = PPO.load("ppo-LunarLander-v3_v2", env=env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")