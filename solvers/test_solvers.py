import gym
import sys 
sys.path.append("..")

import gym_snake

from shortest_path import  ShortestPathBFSSolver

import time 




def test_ai(env,ai):

    for i in range(10):
        env.reset()
        t = 0 
        while True:
            env.render()
            
            action = ai.move(env.game)
            
            time.sleep(0.1)

            observation, reward, done, info = env.step(action)
            t+= 1
            
            if done:
                print('episode {} finished after {} timesteps'.format(i, t))
                env.close()
                break
        
            

if __name__ == "__main__":
    env = gym.make('snake-v0')
    solver = ShortestPathBFSSolver()
    test_ai(env,ai = solver)


