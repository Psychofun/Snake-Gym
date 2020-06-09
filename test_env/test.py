import gym
import sys 
import time
import numpy as np
sys.path.append("..")

import gym_snake



if __name__ == "__main__":
    # TEST 
    
    env = gym.make('snake-v0')
    
    

    for a in [2,3,2,2,2,2,2,2,2,2,2,2,2,2,3,2,2,3]:
        env.render()
        time.sleep(1.0)
        observation, reward, done, info = env.step(a)     
        if done:
            break
    env.close()