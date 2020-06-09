# Snake-Gym
Snake game in python, gym style




Based on [Seanbae](https://github.com/seanbae/gym-snake) and [Gsurma](https://github.com/gsurma/slitherin) repositories.

1) Clone the repo:
```
$ git clone https://github.com/Psychofun/Snake-Gym.git
```

2) `cd` into `snake-gym` and run:
```
$ pip install -e .
```

3) This should run total 100 instances of the `snake-v0` environment for 1000 timesteps, rendering the environment at each step. By default, you should see a window pop up rendering the classic Snake problem:
```python
import gym
import gym_snake

env = gym.make('snake-v0')

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            print('episode {} finished after {} timesteps'.format(i, t))
            break


```
