import gym
import time
import numpy as np

DEBUG = False

for e in range(1):
    env = gym.make('gym_nav2d:nav2dVeryEasyXYControl-v0')
    # env = gym.make('gym_nav2d:nav1dVeryEasyYControl-v0')
    #env = gym.make('gym_nav2d:nav2dEasy-v0')
    # env = gym.make('gym_nav2d:nav2dHard-v0')
    # env = gym.make('gym_nav2d:nav2dVeryHard-v0')
    obs = env.reset()
    cumulated_reward = 0
    i = 0
    done = False
    while not done and i <= 1000:
        i += 1
        #act = env.action_space.sample()
        act = [np.random.random(), .99]
        #print("action", act)
        obs, rew, done, info = env.step(act)     # take a random action
        env.render(mode='human')
        #print(rew)
        if rew > 0:
            print("done")
            break
        # print(env.render(mode='ansi'))
        cumulated_reward += rew
        if DEBUG:
            print(info)
    if DEBUG and done:
        time.sleep(3)

    #print("episode ended with cumulated rew", cumulated_reward, "and done:", done)
    env.close()
