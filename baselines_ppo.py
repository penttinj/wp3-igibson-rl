import logging
import math
import os
import sys
import argparse
from tabnanny import check
from typing import Callable

import igibson
from sympy import root
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
    "-e", "--eval", dest="training", action="store_false", help="flag for running evluation only",
)
parser.add_argument(
    "-s", "--steps", metavar="NUM_STEPS", default=40000, type=int, help="number of steps to train"
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
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size), nn.ReLU()
                )
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
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


def get_latest_ckpt(log_dir, log_name, ckpt_root_dir):
    run_id = get_latest_run_id(log_dir, log_name)
    ckpt_run_dir = os.path.join(ckpt_root_dir, f"{log_name}_{run_id}")
    ckpt_name = f"{log_name}-{run_id}-ckpt"
    os.makedirs(ckpt_run_dir, exist_ok=True)
    return os.path.join(ckpt_run_dir, f"{ckpt_name}_"), get_latest_run_id(ckpt_run_dir, ckpt_name)


def main(training=True, num_steps=80000):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_file = "configs/go_to_object.yaml"
    root_dir = "results_baselines"
    tensorboard_log_dir = os.path.join(root_dir, "logs")
    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    num_environments = 8

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> Wp3TestEnv:
            env = Wp3TestEnv(
                config_file=config_file,
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
        env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
        env = VecMonitor(env)

        # Create a new environment for evaluation
        eval_env = Wp3TestEnv(
            config_file=config_file,
            mode="gui_interactive",
            action_timestep=1 / 10.0,
            physics_timestep=1 / 120.0,
        )

        # Obtain the arguments/parameters for the policy and create the PPO model
        policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=policy_kwargs,
        )
        print(f"{model.policy=}")

        # Create a checkpoint folder for this run
        ckpt_base, ckpt_id = get_latest_ckpt(
            log_dir=tensorboard_log_dir, log_name="PPO", ckpt_root_dir=checkpoint_dir
        )

        # Random Agent, evaluation before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
        for i in range(5):
            # Train the model for the given number of steps
            model.learn(math.floor(num_steps / 5))
            model.save(f"{ckpt_base}{ckpt_id+1}")
            ckpt_id += 1

        # Evaluate the policy after training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

        # Save the trained model and delete it
        model.save(f"{ckpt_base}{ckpt_id+1}")

        # Reload the trained model from file
        model = PPO.load(f"{ckpt_base}{ckpt_id+1}")

        # Evaluate the trained model loaded from file
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    else:
        logging.info("Eval only mode")
        ckpt_base, ckpt_id = get_latest_ckpt(
            log_dir=tensorboard_log_dir, log_name="PPO", ckpt_root_dir=checkpoint_dir
        )
        print("checkpoint=", ckpt_base, ckpt_id)
        eval_env = Wp3TestEnv(
            config_file=config_file,
            mode="gui_interactive",
            action_timestep=1 / 10.0,
            physics_timestep=1 / 120.0,
        )

        model = PPO.load(f"{ckpt_base}{ckpt_id}")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main(training=args.training, num_steps=args.steps)
