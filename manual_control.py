import logging
import math
import os
import sys
import argparse
import time
from typing import Callable, List, Union
from subprocess import run
from Actions import Actions
import curses
from curses import wrapper
import asyncio


try:
    import gym
    import torch as th
    import torch.nn as nn
    from envs.wp3_test_env import Wp3TestEnv
    from igibson.utils.utils import parse_config

except ModuleNotFoundError as e:
    print(f"{e} found. Install it or activate the correct conda environment?")
    exit(1)


def create_action(yaw=0.0, translation=0.0):
    return [translation * 0.3, yaw * 0.3]


def main(stdscr, config_path):
    stdscr.nodelay(True)
    env = Wp3TestEnv(
        config_file=config_path,
        mode="gui_interactive",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )
    state = env.reset()
    # a = {"forward": 1.0, "backward": -1.0, "right": 1.0,
    # "left": -1.0} # make into enum
    #action_space = env.action_space
    start = time.time()
    while True:
        action = ""
        stdscr.refresh()
        key = stdscr.getch()
        if key == ord("w"):
            action = create_action(translation=Actions.FORWARD)
        elif key == ord("s"):
            action = create_action(translation=Actions.BACKWARD)
        elif key == ord("a"):
            action = create_action(yaw=Actions.LEFT)
        elif key == ord("d"):
            action = create_action(yaw=Actions.RIGHT)
        else:
            action = create_action()
        if action == "":
            action = None
        state, reward, done, _ = env.step(action)
        if done:
            stdscr.addstr(0, 0, "Episode has finished running.",
              curses.A_REVERSE)
            stdscr.refresh()
            #break
    print(
       "Episode finished after {} timesteps, took {} seconds.".format(
           env.current_step, time.time() - start
       )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="testing shit", type=str, required=True)

    args = parser.parse_args()
    wrapper(main, config_path=args.config)

