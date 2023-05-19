from gym import error, spaces, utils
import numpy as np
import math

from gym_nav2d.envs.nav2d_env import Nav2dEnv


class Nav2dMediumXPenaltyEnv(Nav2dEnv):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        Nav2dEnv.__init__(self)

    def _calculate_position(self, action):
        action = np.clip(action, self.action_step_low, self.action_step_high)
        self.agent_x = self.agent_x + action[0]
        #print("action", action)
        self.agent_y = self.agent_y + action[1]

        # borders
        if self.agent_x < 0:
            self.agent_x = 0
        if self.agent_x > self.len_court_x:
            self.agent_x = self.len_court_x
        if self.agent_y < 0:
            self.agent_y = 0
        if self.agent_y > self.len_court_y:
            self.agent_y = self.len_court_y

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)
        # calulate new observation
        obs = self._observation()

        # done for rewarding
        done = bool(obs[4] <= self.eps)
        #print("obs[4]", obs[4], "eps", self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
            if self.agent_x < self.goal_x-20 or self.agent_x > self.goal_x+20:
                rew = -40
        else:
            rew += self._reward_goal_reached()

        # break if more than max_steps actions taken
        done = bool(obs[4] <= self.eps or self.count_actions >= self.max_steps)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])

        normalized_obs = self._normalize_observation(obs)

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[0]) + "," + str(action[1]) + ", dist:" + str(normalized_obs[4]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)
        return normalized_obs, rew, done, {"info":info}
    
    def reset(self, goal_x=200, goal_y=200, agent_x=200, agent_y=None):
        # semi random start point and fixed goal point
        self.count_actions = 0
        self.positions = []
        self.agent_x = agent_x
        if agent_y is None:
            if np.random.uniform() < 0.5:
                self.agent_y = 10
            else:
                self.agent_y = 390
        else:
            self.agent_y = agent_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        if self.goal_y == self.agent_y:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale,
                  self.goal_y*self.scale)
        obs = self._observation()
        return self._normalize_observation(obs)
