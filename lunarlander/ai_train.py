import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


def linear_schedule(initial_value):
    # Learning rate goes down linearly from initial_value to 0
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule


# Training environment (continuous version of LunarLander)
env = make_vec_env("LunarLander-v3", n_envs=1, env_kwargs={"continuous": True})

# Separate evaluation environment
eval_env = Monitor(gym.make("LunarLander-v3", continuous=True))

# Create the SAC model with tuned parameters
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(7.3e-4),
    buffer_size=1_000_000,
    batch_size=256,
    ent_coef="auto",
    gamma=0.99,
    tau=0.01,
    train_freq=1,
    gradient_steps=1,
    learning_starts=10_000,
    policy_kwargs=dict(net_arch=[400, 300]),
    verbose=1,
)

# Test the agent every 10k steps and save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_sac",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

# Train for 500k steps
model.learn(total_timesteps=int(5e5), callback=eval_callback)

# Save the last model with a new name
model.save("sac-LunarLander-v3")

# Load the best model and evaluate it with 20 episodes
best_model = SAC.load("best_sac/best_model")
mean_reward, std_reward = evaluate_policy(
    best_model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")