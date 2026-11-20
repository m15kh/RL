import  gymnasium as gym
from collections import defaultdict
import numpy as np 


env = gym.make("FrozenLake-v1")

eps = 0.05
alpha = 0.2
gamma = 1.0

nb_actions = 4



def sarsa(eps, alpha):
    
    Q = defaultdict(float)

    def policy(St, pi):

        return np.random.choise([0, 1, 2, 3], p = [pi[(St, a)] for a in [0, 1, 2, 3]])
        
    pi = defaultdict(lambda : 1/nb_actions)
