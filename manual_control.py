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
from curses import wrapper, window
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

def log(s):
    with open("log.txt", "a") as f:
        f.write(f"{'-'*10}\n")
        f.write(f"[{time.asctime()}] {s}\n")

def main(stdscr: window, config_path: str):
    stdscr.nodelay(True)
    env = Wp3TestEnv(
        config_file=config_path,
        mode="gui_interactive",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )
    action_list = {
        ord("w"): create_action(translation=Actions.FORWARD),
        ord("s"): create_action(translation=Actions.BACKWARD),
        ord("a"): create_action(yaw=Actions.LEFT),
        ord("d"): create_action(yaw=Actions.RIGHT),
    }
    state = env.reset()
    log(state)
    start = time.time()
    
    while True:
        stdscr.refresh()
        key = stdscr.getch()

        if key in action_list.keys():
            action = action_list[key]
        elif key == ord("q"):
            sys.exit(0)
        else:
            action = create_action()

        state, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="testing shit", type=str, required=True)

    args = parser.parse_args()
    wrapper(main, config_path=args.config)

