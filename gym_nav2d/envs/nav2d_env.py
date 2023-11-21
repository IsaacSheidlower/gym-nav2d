import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math, time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.transforms as transforms
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class Nav2dEnv(gym.Env):
    # this is a list of supported rendering modes!
    metadata = {'render.modes': ['human', 'ansi'],
                'video.frames_per_second': 30}

    def __init__(self):
        self.debug = False
        # define the environment and the observations
        self.len_court_x = 400              # the size of the environment
        self.len_court_y = 400              # the size of the environment

        # TO INCLUDE DISTANCE TO GOAL
        # self.obs_low_state = np.array([-1, -1, -1, -1, 0]) # x_agent,y_agent, x_goal, y_goal, distance
        # self.obs_high_state = np.array([1, 1, 1, 1, 1])

        # NO DISTANCE TO GOAL
        self.obs_low_state = np.array([-1, -1, -1, -1]) # x_agent,y_agent, x_goal, y_goal, distance
        self.obs_high_state = np.array([1, 1, 1, 1])
        self.observation_space = spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)

        self.max_steps = 1000
        self.max_step_size = 10
        # action space: change direction in rad (discrete), run into this direction (Box)
        self.action_angle_low = -1
        self.action_angle_high = 1
        self.action_step_low = -1
        self.action_step_high = 1
        self.action_space = spaces.Box(np.array([self.action_angle_low, self.action_step_low]),
                                       np.array([self.action_angle_high, self.action_step_high]), dtype=np.float32)

        self.count_actions = 0  # count actions for rewarding
        self.eps = 3  # distance to goal, that has to be reached to solve env
        self.np_random = None  # random generator

        # agent
        self.agent_x = 0
        self.agent_y = 0
        self.positions = []                 # track agent positions for drawing

        # the goal
        self.goal_x = 0
        self.goal_y = 0

        # rendering
        self.screen_height = 415
        self.screen_width = 400
        self.viewer = None                  # viewer for render()
        self.agent_trans = None             # Transform-object of the moving agent
        self.track_way = None               # polyline object to draw the tracked way
        self.scale = self.screen_width/self.len_court_x 
        
        # set self.last_action to 0,0
        self.last_action = np.array([0, 0])

        # set a seed and reset the environment
        self.seed()
        self.reset()

    def _distance(self):
        return math.sqrt(pow((self.goal_x - self.agent_x), 2) + pow(self.goal_y - self.agent_y, 2))

    # todo: think about a good reward fct that lets the agents learn to go to the goal by
    #  extra rewarding reaching the goal and learning to do this by few steps as possible
    def _reward_goal_reached(self):
        # 1000 - (distance)/10 - (sum of actions)
        return 100

    def _step_reward(self):
        return - self._distance()/10 - 1

    def _observation(self):
        return np.array([self.agent_x, self.agent_y, self.goal_x, self.goal_y, self._distance()])

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, 4):
            normalized_obs.append(obs[i]/255*2-1)
        normalized_obs.append(obs[-1]/360.62)
        return np.array(normalized_obs)

    def _calculate_position(self, action):
        if not self.xy_control:
            angle = (action[0] + 1) * math.pi + math.pi / 2
            if angle > 2 * math.pi:
                angle -= 2 * math.pi
            step_size = (action[1] + 1) / 2 * self.max_step_size
            # calculate new agent state
            self.agent_x = self.agent_x + math.cos(angle) * step_size
            self.agent_y = self.agent_y + math.sin(angle) * step_size
        else:
            # clip actions
            action = np.clip(action, self.action_step_low, self.action_step_high)
            self.agent_x = self.agent_x + action[0]
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
        rew = 0
        if not done:
            rew += self._step_reward()
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

        self.last_action = action

        return normalized_obs, rew, done, {"info":info}

    def reset(self):
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        # self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.agent_x = 10
        self.agent_y = 240
        # self.goal_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.goal_y = self.np_random.uniform(low=0, high=self.len_court_x)
        self.goal_x = 125
        self.goal_y = 125
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
        self.last_action = np.array([0, 0])
        return self._normalize_observation(obs)

    def render(self, mode='human', agent_color='b', subgoals=None, subgoal_colors=None, trace=True):
        if mode == 'ansi':
            return self._observation()
        elif mode == 'human' or mode == 'rgb':

            # create figure
            if self.viewer is None:
                self.viewer = plt.figure(figsize=(6, 6))
                #self.viewer.canvas.set_window_title("GoalEnv")
                self.ax = self.viewer.add_subplot(111)
                self.ax.set_xlim(0, self.screen_width)
                self.ax.set_ylim(0, self.screen_height)
                self.ax.set_aspect('equal')
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_title("GoalEnv")
                self.ax.set_xlabel("x")
                self.ax.set_ylabel("y")
                self.ax.grid(True)

                # # create court
                # self.ax.add_patch(
                #     patches.Rectangle(
                #         (0, 0), self.len_court_x, self.len_court_y, fill=False, linewidth=1, edgecolor='black'
                #     )
                # )


                # create goal
                self.goal = patches.Circle((self.goal_x, self.goal_y), 5, fc='r', zorder=10)
                self.ax.add_patch(self.goal)

                # create agent
                self.agent = patches.Circle((self.agent_x, self.agent_y), 5, fc=agent_color, alpha=1)
                # self.ax.add_patch(self.agent)
                # render agent as triangle (arrow) facing the direction of the last action
                # the last action is of the form [x,y] with x,y in [-1,1]
                # self.agent = patches.RegularPolygon((self.agent_x, self.agent_y), 3, 10, np.arctan2(self.last_action[1], self.last_action[0])-np.pi/2, fc='b')
                # self.ax.add_patch(self.agent)

                # shade areas x>180 and x<220   
                self.ax.add_patch(
                    patches.Rectangle(
                        (180, 0), 40, 400, fill=True, linewidth=1, edgecolor='black', facecolor='grey', alpha=0.5
                    )
                )
                
                # create line
                self.line, = self.ax.plot([], [], 'b-')

                # load car image
                curr_path = os.path.dirname(os.path.abspath(__file__))
                self.robot_img = plt.imread(curr_path + "/robot.png")
                self.robot_size = 15
                self.robot = self.ax.imshow(self.robot_img, extent=[self.agent_x-self.robot_size, self.agent_x+self.robot_size, self.agent_y-self.robot_size, self.agent_y+self.robot_size], alpha=1, zorder=10)

                if subgoals is not None:
                    for index, subgoal in enumerate(subgoals):
                        print("subgoal", subgoal)
                        # make circular subgoals
                        self.ax.add_patch(
                            patches.Circle(
                                (subgoal[0], subgoal[1]), 5, fc=subgoal_colors[index])
                        )
                        
            # show figure
            # leave a trace of the agent
            # with high color saturation
            if trace:
                new_agent = patches.Circle((self.agent_x, self.agent_y), 5, fc=agent_color, alpha=0.1)
                self.ax.add_patch(new_agent)
                self.agent.center = (self.agent_x, self.agent_y)
                # self.ax.add_patch(self.agent)
            # update the robot image
            self.robot.set_extent([self.agent_x-self.robot_size, self.agent_x+self.robot_size, self.agent_y-self.robot_size, self.agent_y+self.robot_size])

            # self.agent = patches.RegularPolygon((self.agent_x, self.agent_y), 3, 10, np.arctan2(self.last_action[1], self.last_action[0])-np.pi/2, fc='b', alpha=0.01)
            # self.ax.add_patch(self.agent)
            # new_angle = np.arctan2(self.last_action[1], self.last_action[0]) - np.pi/2
            # # smooth angle change
            # new_angle = (new_angle + self.agent.orientation)/2
            # self.agent = patches.RegularPolygon((self.agent_x, self.agent_y), 3, 10, new_angle, fc='b')
            # self.ax.add_patch(self.agent)

            self.goal.center = (self.goal_x, self.goal_y)
            #self.line.set_data(*zip(*self.positions))
            if mode == 'rgb':
                # Convert the matplotlib plot to an RGB array
                self.viewer.canvas.draw()
                width, height = self.viewer.get_size_inches() * self.viewer.get_dpi()
                mpl_image = np.frombuffer(self.viewer.canvas.tostring_rgb(), dtype='uint8')
                mpl_image = mpl_image.reshape(int(height), int(width), 3)
                return mpl_image

            self.viewer.canvas.draw()
            plt.pause(0.0001)
            return self.viewer
        



    def close(self):
        # if self.viewer:
        #     self.viewer.close()
        #     self.viewer = None
        pass
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
