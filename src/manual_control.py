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
    from envs.dynamic_env import DynamicEnv
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
    env = DynamicEnv(
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
    stdscr.addstr(0, 0, "Input ready. Use WASD for steering, Q to quit.", curses.A_REVERSE)
    stdscr.refresh()

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

        max_y, max_x = stdscr.getmaxyx()
        stdscr.addstr(1, 0, " "*max_x)
        stdscr.addstr(2, 0, " "*max_x)
        stdscr.addstr(3, 0, " "*max_x)
        stdscr.addstr(1, 0, f"Rewards: {reward}")
        stdscr.addstr(2, 0, f"Info: {_}")
        stdscr.addstr(3, 0, f"Waypoints obs: {state['waypoints']}")
        stdscr.addstr(max_y-1, 0, f"Screen size: {max_x} x {max_y}")
        stdscr.refresh()
        if done:
            stdscr.addstr(
                4,
                0,
                f"Episode done.\nEpisode finished after {env.current_step} timesteps\
, took {time.time() - start} seconds.\nPress Q to quit or continue exploring",
            )
            stdscr.refresh()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="testing shit", type=str, required=True)

    args = parser.parse_args()
    wrapper(main, config_path=args.config)

