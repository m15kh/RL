import gymnasium as gym
import time
from gymnasium.envs.toy_text.frozen_lake import LEFT, DOWN, RIGHT, UP
from SmartAITool.core import cprint

env = gym.make('FrozenLake-v1', render_mode="human")
map_actions = {0:"Left",1:"Down",2:"Right",3:"Up"}
episode = 20
print(env.unwrapped)
print(env.observation_space)
print(env.action_space)
print(list(range(env.action_space.n)))




for i in range(episode):
    
    env.reset()
    env.render()
    cprint(f"episode {i +1 }",'blue')
    while True:
        random_action = env.action_space.sample()
        # out = env.step(random_action)
        next_stage, reward, terminated, truncated, info = env.step(random_action)
        print(map_actions[random_action])
        print(next_stage, reward, terminated, truncated, info)
        if reward != 0:
            cprint(reward)
        if terminated:
            break


time.sleep(1)  
env.close()