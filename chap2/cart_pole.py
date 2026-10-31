import gymnasium as gym
import time
from SmartAITool.core import cprint


env = gym.make('CartPole-v1', render_mode="human")

print(env.observation_space)


for i in range(20):
    print(f"episode: {i+1}")
    obs = env.reset()
    env.render()

    while True:
        random_action = env.action_space.sample()
        # out = env.step(random_action)
        next_stage, reward, terminated, truncated, info = env.step(random_action)
        print(next_stage, reward, terminated, truncated, info)
        if reward != 0:
            cprint(reward)
        if terminated:
            break


time.sleep(1)