import logging
import math
import os
import sys
import argparse
from tabnanny import check
import time
from typing import Callable

import igibson
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


def get_latest_ckpt(log_dir: str, log_name: str, ckpt_root_dir: str):
    run_id = get_latest_run_id(log_dir, log_name)
    ckpt_run_dir = os.path.join(ckpt_root_dir, f"{log_name}_{run_id}")
    ckpt_name = f"{log_name}-{run_id}-ckpt"
    os.makedirs(ckpt_run_dir, exist_ok=True)
    return os.path.join(ckpt_run_dir, f"{ckpt_name}_"), get_latest_run_id(ckpt_run_dir, ckpt_name)


def main(training: bool = True, num_steps: int = 80000, checkpoint_interval: int = 10000):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    assert (
        math.floor(num_steps / checkpoint_interval) > 0
    ), "Number of steps must be larger than checkpoint interval"
    config_file = "configs/go_to_object.yaml"
    root_dir = "results_baselines"
    tensorboard_log_dir = os.path.join(root_dir, "logs")
    checkpoint_dir = os.path.join(
        root_dir, "checkpoints", f"PPO_{get_latest_run_id(os.path.join(root_dir, 'logs'), 'PPO')+1}"
    )
    num_environments = 8
    checkpoint_freq = checkpoint_interval // num_environments
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=checkpoint_dir, name_prefix="ppo_model",
    )

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
        eval_env = SubprocVecEnv(
            [
                lambda: Wp3TestEnv(
                    config_file=config_file,
                    mode="gui_interactive",
                    action_timestep=1 / 10.0,
                    physics_timestep=1 / 120.0,
                )
            ]
        )
        eval_env = VecMonitor(eval_env)

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
            batch_size=64,
        )
        print(f"{model.policy=}")

        # Random Agent, evaluation before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
        start = time.time()
        # Train the model for the given number of steps
        model.learn(num_steps, callback=checkpoint_cb)

        # Evaluate the policy after training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
        end = time.time()
        train_time = end - start

        logging.info("Time taken for training ", train_time)
    else:
        logging.info("Eval only mode")
        ckpt_base, ckpt_id = get_latest_ckpt(
            log_dir=tensorboard_log_dir, log_name="PPO", ckpt_root_dir=checkpoint_dir
        )
        print("checkpoint=", ckpt_base, ckpt_id)
        eval_env = SubprocVecEnv(
            [
                lambda: Wp3TestEnv(
                    config_file=config_file,
                    mode="gui_interactive",
                    action_timestep=1 / 10.0,
                    physics_timestep=1 / 120.0,
                )
            ]
        )
        eval_env = VecMonitor(eval_env)
        # model = PPO.load(f"{ckpt_base}{ckpt_id}")
        model = PPO.load(f"results_baselines/checkpoints/PPO_30/PPO-30-ckpt_2.zip")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main(training=args.training, num_steps=args.steps, checkpoint_interval=args.checkpoint_interval)

