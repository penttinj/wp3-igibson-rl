#! /usr/bin/env python3

import logging
import math
import os
import sys
import argparse
import time
from typing import Callable, List, Union
from subprocess import run

from igibson.utils.utils import parse_config

from envs.wp3_test_env import Wp3TestEnv

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed, get_latest_run_id
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)
"""
Example training code using stable-baselines3 PPO for PointNav task.
"""

parser = argparse.ArgumentParser(
    description="Train a Turtlebot in an iGibson environment using PyTorch stable-baselines3's PPO"
)
parser.add_argument(
    "-e", "--eval", dest="training", action="store_false", help="flag for running evaluation only",
)
parser.add_argument(
    "-s", "--steps", metavar="NUM_STEPS", default=40000, type=int, help="number of steps to train"
)
parser.add_argument(
    "-c",
    "--checkpoint_interval",
    default=10000,
    type=int,
    help="number of steps interval between checkpoints",
)
parser.add_argument(
    "-n", "--num_envs", default=8, type=int, help="number of parallel environments",
)
parser.add_argument(
    "-m", "--model", type=str, help="path to a saved model(.zip file)",
)
parser.add_argument(
    "--config", default="configs/go_to_object.yaml", type=str, help="path to yaml config file"
)
args = parser.parse_args()


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 256
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs", "waypoints"]:
                print(f"CustomCombinedExtractor: {key}, {subspace.shape[0]=}")
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size), nn.ReLU()
                )
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                print(
                    f"CustomCombinedExtractor: {key},(num input channels): {subspace.shape[2]=}, {subspace.shape=}"
                )
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                print(
                    f"CustomCombinedExtractor: {key},(num input channels): {subspace.shape[1]=}, {subspace.shape=}"
                )
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def get_latest_ckpt(log_dir: str, log_name: str, ckpt_root_dir: str):
    run_id = get_latest_run_id(log_dir, log_name)
    ckpt_run_dir = os.path.join(ckpt_root_dir, f"{log_name}_{run_id}")
    ckpt_name = f"{log_name}-{run_id}-ckpt"
    os.makedirs(ckpt_run_dir, exist_ok=True)
    return os.path.join(ckpt_run_dir, f"{ckpt_name}_"), get_latest_run_id(ckpt_run_dir, ckpt_name)


def get_hyperparams(config_path: str):
    config = parse_config(config_path)
    data = {
        "num_envs": config.get("num_envs", args.num_envs),
    }
    # Assignment expression syntax https://stackoverflow.com/a/2604036
    if batch_size := config.get("batch_size"):
        data["batch_size"] = batch_size
    if gamma := config.get("gamma"):
        data["gamma"] = gamma
    if learning_rate := config.get("learning_rate"):
        data["learning_rate"] = learning_rate
    return data


def main(
    config_file: str,
    hyperparameters,
    simulation_scenes=None,
    num_envs=6,
    training: bool = True,
    num_steps: int = 80000,
    checkpoint_interval: int = 20000,
):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    assert (
        math.floor(num_steps / checkpoint_interval) > 0
    ), "Number of steps must be larger than checkpoint interval"

    root_dir = "results_baselines"
    tensorboard_log_dir = os.path.join(root_dir, "logs")
    checkpoint_dir = os.path.join(
        root_dir,
        "checkpoints",
        f"PPO_{get_latest_run_id(os.path.join(root_dir, 'logs'), log_name='PPO')+1}",
    )
    checkpoint_freq = checkpoint_interval // num_envs
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=checkpoint_dir, name_prefix="ppo_model",
    )

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0, scenes: Union[List[str], None]=None) -> Callable:
        def _init() -> Wp3TestEnv:
            scene = None if scenes is None else scenes[rank % len(scenes)]
            env = Wp3TestEnv(
                config_file=config_file,
                scene_id=scene,
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
                device_idx=0,
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    if training:
        # Multiprocess
        env = SubprocVecEnv([make_env(rank=i, scenes=simulation_scenes) for i in range(1, num_envs + 1)])
        env = VecMonitor(env)
        # Create a new environment for evaluation
        eval_env = SubprocVecEnv(
            [
                lambda: Wp3TestEnv(
                    config_file=config_file,
                    mode="headless",
                    action_timestep=1 / 10.0,
                    physics_timestep=1 / 120.0,
                    device_idx=0,
                )
            ]
        )
        eval_env = VecMonitor(eval_env)

        # Obtain the arguments/parameters for the policy and create the PPO model
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
            net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])],
        )
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        run(f"cp {config_file} {checkpoint_dir}", shell=True)

        model = (
            PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                policy_kwargs=policy_kwargs,
                **hyperparameters,
            )
            if args.model is None
            else PPO.load(
                args.model, env, verbose=1, tensorboard_log=tensorboard_log_dir, batch_size=256,
            )
        )
        
        # Random Agent, evaluation before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
        print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

        start = time.time()
        # Train the model for the given number of steps
        model.learn(num_steps, callback=checkpoint_cb)
        # Evaluate the policy after training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
        end = time.time()
        train_time = end - start

        logging.info(f"Time taken for training {train_time} seconds")
    else:
        logging.info("Eval only mode")
        logging.info(f"Using config {config_file}")
        logging.info(f"Using model {args.model}")
        eval_env = Wp3TestEnv(
            config_file=config_file,
            mode="gui_interactive",
            action_timestep=1 / 10.0,
            physics_timestep=1 / 120.0,
        )
        # Reset camera position to the middle of the scene
        s = eval_env.simulator
        s.viewer.initial_pos = [0, 0, 1]
        s.viewer.initial_view_direction = [0.6, -0.8, -0.3]
        s.viewer.reset_viewer()
        eval_env = Monitor(eval_env)
        model = PPO.load(args.model)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=40)
        print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = parse_config(args.config)
    hyperparams = get_hyperparams(args.config)
    num_envs = hyperparams["num_envs"]
    del hyperparams["num_envs"]
    print(f"{num_envs=}")
    scenes = config.get("simulation_scenes", None)
    main(
        config_file=args.config,
        training=args.training,
        num_envs=num_envs,
        num_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        hyperparameters=hyperparams,
        simulation_scenes=scenes,
    )
