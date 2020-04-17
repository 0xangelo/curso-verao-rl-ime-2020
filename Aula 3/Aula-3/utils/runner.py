import math
import time

import numpy as np


def run_experiments(n_trials, env, agent_cls, config, postprocessing, total_timesteps):
    results = {
        "timesteps": [],
        "total_rewards": [],
        "avg_total_rewards": [],
    }

    for trial in range(n_trials):
        print(f">> Trial {trial + 1} ...")
        agent = agent_cls(env.observation_space, env.action_space, config, postprocessing=postprocessing)
        timesteps, total_rewards, avg_total_rewards = train(agent, env, total_timesteps)
        print()
        results["timesteps"].append(timesteps)
        results["total_rewards"].append(total_rewards)
        results["avg_total_rewards"].append(avg_total_rewards)

    x = np.unique(np.sort(np.concatenate(results["timesteps"])))

    def interp(x, timesteps, metric):
        ys = []
        for trial in range(n_trials):
            xp = timesteps[trial]
            fp = metric[trial]
            left = right = np.nan
            y = np.interp(x, xp, fp, left, right)
            ys.append(y)
        return np.array(ys)

    total_rewards = interp(x, results["timesteps"], results["total_rewards"])
    total_rewards_mean = np.mean(total_rewards, axis=0)
    total_rewards_std = np.std(total_rewards, axis=0)

    avg_total_rewards = interp(x, results["timesteps"], results["avg_total_rewards"])
    avg_total_rewards_mean = np.mean(avg_total_rewards, axis=0)
    avg_total_rewards_std = np.std(avg_total_rewards, axis=0)

    return x, (total_rewards_mean, total_rewards_std), (avg_total_rewards_mean, avg_total_rewards_std)


def train(agent, env, total_timesteps):
    timesteps = []
    total_rewards = []
    avg_total_rewards = []

    timestep = 0
    episode = 0

    start_time = time.time()

    while timestep < total_timesteps:
        total_reward = 0.0
        episode_length = 0

        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _  = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)

            total_reward += reward
            episode_length += 1
            timestep += 1

            obs = next_obs

        timesteps.append(timestep)
        total_rewards.append(total_reward)
        avg_total_rewards.append(np.mean(total_rewards[-100:]))

        episode += 1

        loss = agent.learn()

        if loss is not None:
            if isinstance(loss, tuple):
                policy_loss, value_loss = loss
                loss_str = f"policy_loss = {policy_loss:12.4f}, value_loss = {value_loss:10.4f}"
            else:
                loss_str = f"loss = {loss:12.4f}"

            ratio = math.ceil(100 * timestep / total_timesteps)
            uptime = math.ceil(time.time() - start_time)

            print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d} -> {loss_str}, total_reward = {total_reward:10.4f}, episode_length = {episode_length:3d}\r", end="")

    return timesteps, total_rewards, avg_total_rewards


def evaluate(agent, env, n_episodes, render=False):
    
    timesteps = []
    total_rewards = []
    avg_total_rewards = []

    timestep = 0
    episode = 0

    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        total_reward = 0.0
        episode_length = 0

        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _  = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)

            total_reward += reward
            episode_length += 1
            timestep += 1

            obs = next_obs
            
            if render:
                env.render(mode="human")

        if render:
            env.close()

        timesteps.append(timestep)
        total_rewards.append(total_reward)
        avg_total_rewards.append(np.mean(total_rewards[-100:]))

        ratio = math.ceil(100 * episode / n_episodes)
        uptime = math.ceil(time.time() - start_time)
        print(f"[{ratio:3d}% / {uptime:3d}s] episode = {episode}/{n_episodes} -> total_reward = {total_reward:10.4f}, episode_length = {episode_length:3d}\r", end="")

    return timesteps, total_rewards, avg_total_rewards
