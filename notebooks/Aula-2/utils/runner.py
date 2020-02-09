import time

import numpy as np


def train(agent, env, total_timesteps, verbose=True):
    timesteps = []
    losses = []
    grads = []
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

        result = agent.learn()

        if verbose and result is not None:
            loss, gradients = result

            losses.append((timestep, loss))
            grads.append((timestep, gradients))

            ratio = int(100 * timestep / total_timesteps)
            uptime = int(time.time() - start_time)
            print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d} -> loss = {loss:10.4f}, total_reward = {total_reward:10.4f}, episode_length = {episode_length:3d}\r", end="")

    print()

    return timesteps, losses, grads, total_rewards, avg_total_rewards


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

        ratio = int(100 * episode / n_episodes)
        uptime = int(time.time() - start_time)
        print(f"[{ratio:3d}% / {uptime:3d}s] episode = {episode}/{n_episodes} -> total_reward = {total_reward:10.4f}, episode_length = {episode_length:3d}\r", end="")

    print()

    return timesteps, total_rewards, avg_total_rewards