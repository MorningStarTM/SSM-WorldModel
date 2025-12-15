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
from ssm.utils.converter import ActionConverter
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, WeightedRandomOpponent, BaseActionBudget
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



    def attack_opponent_data_generation(
            self,
            start=0,
            folder_name="ssm\\data_generation\\attack_data",
            opponent_kind="weighted",            # "random" or "weighted"
            file_name="attack_dataset.npy",
            num_times=5,                         # <--- NEW: how many full passes
            lines_attacked=None,
            rho_normalization=None,
            opponent_attack_cooldown=12*24,
            opponent_attack_duration=12*4,
            opponent_budget_per_ts=0.5,
            opponent_init_budget=0.0,
            seed=0,
            keep_only_failures=False,
        ):
        env_name = self.config["env_name"]

        if opponent_kind.lower() == "random":
            opp_cls = RandomLineOpponent
        else:
            opp_cls = WeightedRandomOpponent

        # get all line names once
        tmp_env = grid2op.make(env_name, reward_class=L2RPNSandBoxScore, backend=LightSimBackend())
        if lines_attacked is None:
            lines_attacked = list(tmp_env.name_line)
        tmp_env.close()

        kwargs_opponent = {"lines_attacked": lines_attacked}

        if opp_cls is WeightedRandomOpponent:
            kwargs_opponent["attack_period"] = opponent_attack_cooldown

            if rho_normalization is None:
                kwargs_opponent["rho_normalization"] = np.ones(len(lines_attacked), dtype=np.float32)
            else:
                rn = np.asarray(rho_normalization, dtype=np.float32)
                if rn.ndim == 0:
                    rn = np.full(len(lines_attacked), float(rn), dtype=np.float32)
                kwargs_opponent["rho_normalization"] = rn

        # --------- GLOBAL storage across ALL runs ----------
        all_episodes = []

        # --------- OUTER LOOP ----------
        for run_id in range(num_times):
            logger.info(f"[ATTACK DATA] ===== RUN {run_id+1}/{num_times} =====")

            # create a fresh env per run (clean + reproducible)
            env = grid2op.make(
                env_name,
                reward_class=L2RPNSandBoxScore,
                backend=LightSimBackend(),
                opponent_attack_cooldown=opponent_attack_cooldown,
                opponent_attack_duration=opponent_attack_duration,
                opponent_budget_per_ts=opponent_budget_per_ts,
                opponent_init_budget=opponent_init_budget,
                opponent_action_class=PowerlineSetAction,
                opponent_class=opp_cls,
                opponent_budget_class=BaseActionBudget,
                kwargs_opponent=kwargs_opponent,
            )

            action_converter = ActionConverter(env)
            random_topology = RandomTopologyAgent(action_converter.actions)

            # seed changes per run (so attacks differ each run)
            run_seed = seed + run_id
            try:
                env.seed(run_seed)
            except Exception:
                pass
            random.seed(run_seed)
            np.random.seed(run_seed)

            num_episodes = len(env.chronics_handler.subpaths)

            # --------- INNER LOOP (episodes) ----------
            for episode_id in range(start, num_episodes + 1):
                logger.info(f"[ATTACK DATA] Run={run_id} Episode={episode_id}")
                env.set_id(episode_id)
                obs = env.reset()

                steps = []
                obs_data = []
                action_data = []
                obs_next_data = []
                reward_data = []
                done_data = []
                info_data = []
                rho_max_data = []

                episode_failed = False

                for t in range(env.max_episode_duration()):
                    try:
                        action = random_topology.act()
                        action_idx = action_converter.action_idx(action)

                        obs_, reward, done, info = env.step(action)

                        obs_data.append(obs.to_vect())
                        action_data.append(action_idx)
                        obs_next_data.append(obs_.to_vect())
                        reward_data.append(reward)
                        done_data.append(done)
                        steps.append(t)

                        info_data.append(info)
                        try:
                            rho_max_data.append(float(np.max(obs_.rho)))
                        except Exception:
                            rho_max_data.append(None)

                        if done and obs_.current_step < obs_.max_step:
                            episode_failed = True
                        try:
                            if (obs_.rho >= 1.0).any():
                                episode_failed = True
                        except Exception:
                            pass

                        obs = obs_
                        if done:
                            break

                    except Grid2OpException as e:
                        logger.info(f"[ATTACK DATA] Grid2OpException at step {t} in episode {episode_id}: {e}")
                        episode_failed = True
                        break

                if keep_only_failures and not episode_failed:
                    continue

                ep = {
                    "run_id": run_id,
                    "seed": run_seed,
                    "episode_id": episode_id,

                    "obs_data": obs_data,
                    "action_data": action_data,
                    "obs_next_data": obs_next_data,
                    "reward_data": reward_data,
                    "done_data": done_data,
                    "steps": steps,

                    "info_data": info_data,
                    "rho_max_data": rho_max_data,
                    "episode_failed": episode_failed,
                }

                all_episodes.append(ep)

            env.close()

        # --------- SAVE ONCE ----------
        dataset = {
            "episodes": all_episodes,
            "meta": {
                "env_name": env_name,
                "opponent_kind": opponent_kind,
                "opponent_attack_cooldown": opponent_attack_cooldown,
                "opponent_attack_duration": opponent_attack_duration,
                "opponent_budget_per_ts": opponent_budget_per_ts,
                "opponent_init_budget": opponent_init_budget,
                "lines_attacked": lines_attacked,
                "base_seed": seed,
                "num_times": num_times,
            }
        }

        os.makedirs(folder_name, exist_ok=True)
        out_path = os.path.join(folder_name, file_name)
        np.save(out_path, dataset, allow_pickle=True)
        logger.info(f"Saved FULL dataset (episodes={len(all_episodes)}) to {out_path}")

        return out_path
