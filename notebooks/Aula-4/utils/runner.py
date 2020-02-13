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
    total_rewards = [[] for _ in range(env.num_envs)]
    avg_total_rewards = []

    total_reward = np.zeros(env.num_envs)
    observations = env.reset()
    timestep = 0
    episode = 0

    t = 0

    start_time = time.time()

    while timestep < total_timesteps:
        actions = agent.act(observations)
        next_observations, rewards, dones, _ = env.step(actions)
        agent.observe(observations, actions, rewards, next_observations, dones)

        loss = agent.learn()

        timestep += len(observations)
        timesteps.append(t)
        t += 1

        total_reward += rewards

        for i in range(env.num_envs):
            if dones[i]:
                total_rewards[i].append((t, timestep, total_reward[i]))
                episode += 1

        if any(G for G in total_rewards):
            episode_returns = sorted(
                list(np.concatenate([G for G in total_rewards if G])),
                key=lambda x: x[1]
            )

            avg_total_rewards.append(
                (t, timestep, np.mean([G[-1] for G in episode_returns[-100:]]))
            )

        total_reward *= 1 - dones
        observations = next_observations

        if loss is not None:
            policy_loss, value_loss, entropy_loss = loss
            loss_str = f"policy_loss = {policy_loss:10.4f}, value_loss = {value_loss:10.4f}, entropy_loss = {entropy_loss:10.4f}"

            ratio = math.ceil(100 * timestep / total_timesteps)
            uptime = math.ceil(time.time() - start_time)

            avg_return = avg_total_rewards[-1][-1] if avg_total_rewards else np.nan

            print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d} -> loss = {loss_str}, avg_return = {avg_return:10.4f}\r", end="")

    return np.array(timesteps), total_rewards, avg_total_rewards


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



if __name__ == "__main__":
    from pprint import pprint
    import gym
    from memory import OnPolicyReplay

    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space
            self.memory = OnPolicyReplay()

        def act(self, obs):
            return self.action_space.sample()

        def observe(self, obs, action, reward, next_obs, done):
            self.memory.update(obs, action, reward, next_obs, done)

        def learn(self):
            return (np.random.normal() for _ in range(3))


    total_timesteps = 500
    num_envs = 5

    env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=True)
    agent = RandomAgent(env.action_space)

    timesteps, total_rewards, avg_total_rewards = train(agent, env, total_timesteps)

#     pprint(timesteps)
#     print()
#     pprint(total_rewards)

#     print()
#     pprint(sorted(list(np.concatenate(total_rewards)), key=lambda x: x[1]))
#     pprint(list(np.sort(np.concatenate(total_rewards), axis=1)))
#     pprint(avg_total_rewards)