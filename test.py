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

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make("LunarLander-v2")
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


env = SubprocVecEnv([make_env(i) for i in range(4)], start_method="fork")
eval_env = Monitor(gym.make("LunarLander-v2"))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)
print("Finished learning")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

for j in range(30):
    obs = eval_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        eval_env.render()
    print("done,", j)
