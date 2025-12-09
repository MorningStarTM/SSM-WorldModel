import os
import dill
import numpy as np
import pandas as pd
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Agent import TopologyGreedy
from grid2op.Reward import L2RPNSandBoxScore
from grid2op.Exceptions import *
from ssm.utils.agent import RandomTopologyAgent
import time
import random
from grid2op.Exceptions import NoForecastAvailable
from collections import defaultdict
from ssm.utils.logger import logger
import warnings
import cloudpickle
from ssm.utils.converter import ActionConverter
warnings.filterwarnings('ignore')




class DataGeneration:
    def __init__(self, config) -> None:
        self.env = grid2op.make(config['env_name'], reward_class=L2RPNSandBoxScore,
                                backend=LightSimBackend())
        self.config = config
        self.action_converter = ActionConverter(self.env)
        self.agent = TopologyGreedy(self.env.action_space)
        self.random_topology = RandomTopologyAgent(self.action_converter.actions)
        #self.agent = TopologyRandom(self.env)
        num = len(self.env.chronics_handler.subpaths)


    def save_trajectories(self, data, folder_name, episode_id):
        """
        Save the trajectory data for a specific episode using dill.

        Args:
            data (dict): A dictionary containing trajectory data (obs_data, action_data, etc.).
            folder_name (str): Directory where the data will be saved.
            episode_id (int): Episode ID to differentiate saved files.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_path = os.path.join(folder_name, f"episode_{episode_id}.pkl")
        
        with open(file_path, 'wb') as f:
            dill.dump(data, f)
        
        logger.info(f"Saved trajectories for episode {episode_id} at {file_path}")





    def load_trajectory(self, file_path):
        """
        Load a trajectory data file.

        Args:
            file_path (str): Path to the saved trajectory file.

        Returns:
            dict: Loaded trajectory data.
        """
        with open(file_path, 'rb') as f:
            return dill.load(f)
        


    def random_topology_data_generation_as_method(self, start=0, folder_name="ssm\\data_generation\\data"):
        num_episodes = len(self.env.chronics_handler.subpaths)

        for episode_id in range(start, num_episodes+1):
            logger.info(f"Episode ID : {episode_id}")
            self.env.set_id(episode_id)
            obs = self.env.reset()

            # Initialize trajectory storage for this episode
            steps = []
            obs_data = []
            action_data = []
            obs_next_data = []
            reward_data = []
            done_data = []

            for i in range(self.env.max_episode_duration()):
                try:
                    action = self.random_topology.act()  # self.agent.act() #
                    obs_, reward, done, _ = self.env.step(action)
                    action_idx = self.action_converter.action_idx(action)

                    # Append data for this step
                    obs_data.append(obs)
                    action_data.append(action_idx)
                    obs_next_data.append(obs_)
                    reward_data.append(reward)
                    done_data.append(done)
                    steps.append(i)

                    # Update observation for the next step
                    obs = obs_

                    if done:
                        self.env.set_id(episode_id)
                        obs = self.env.reset()
                        self.env.fast_forward_chronics(i - 1)
                        
                        action = self.random_topology.act()
                        obs_, reward, done, _ = self.env.step(action)
                        action_idx = self.action_converter.action_idx(action)

                        obs_data.append(obs)
                        action_data.append(action_idx)
                        obs_next_data.append(obs_)
                        reward_data.append(reward)
                        done_data.append(done)
                        steps.append(i)

                        obs = obs_
                except Grid2OpException as e:
                    logger.info(f"Grid2OpException encountered at step {i} in episode {episode_id}: {e}")
                    self.env.set_id(episode_id)
                    obs = self.env.reset()
                    self.env.fast_forward_chronics(i)
                    continue

            # Save trajectory data for this episode
            trajectory_data = {
                "obs_data": obs_data,
                "action_data": action_data,
                "obs_next_data": obs_next_data,
                "reward_data": reward_data,
                "done_data": done_data,
                "steps": steps
            }
            
            self.save_trajectories(trajectory_data, folder_name, episode_id)