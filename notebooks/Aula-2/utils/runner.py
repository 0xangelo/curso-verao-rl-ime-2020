import time

import numpy as np


def run(agent, env, total_timesteps):
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
            ratio = int(100 * timestep / total_timesteps)
            uptime = int(time.time() - start_time)
            print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d} -> loss = {loss:10.4f}, total_reward = {total_reward:10.4f}, episode_length = {episode_length:3d}\r", end="")

    return timesteps, total_rewards, avg_total_rewards
