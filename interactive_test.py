import logging
import os
from sys import platform

import yaml

import igibson
from igibson.envs.wp3_test_env import Wp3TestEnv
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets, download_demo_data


def main():
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs (not interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # If they have not been downloaded before, download assets and Rs Gibson (non-interactive) models
    download_assets()
    download_demo_data()
    config_filename = os.path.join("/home/johan/Projects/WP3/tf2/configs/go_to_object.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    env = Wp3TestEnv(config_file=config_data, mode="gui_interactive")
    # for j in range(10):
    while True:
        logging.info("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if done:
                    logging.info("Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
