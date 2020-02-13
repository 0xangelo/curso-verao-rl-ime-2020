import numpy as np


class OnPolicyReplay:

    def __init__(self):
        self.data_keys = ["states", "actions", "rewards", "next_states", "dones"]
        self.total_experiences = 0
        self.reset()

    def reset(self):
        for key in self.data_keys:
            setattr(self, key, [])

        self.batch_size = 0

    def update(self, states, actions, rewards, next_states, dones):
        transition = (states, actions, rewards, next_states, dones)

        for i, key in enumerate(self.data_keys):
            getattr(self, key).append(np.array(transition[i]))

        self.total_experiences += len(states)
        self.batch_size += len(states)

    def sample(self):
        batch = {key: np.stack(getattr(self, key)) for key in self.data_keys}
        self.reset()
        return batch


if __name__ == "__main__":
    import gym
    from pprint import pprint

    num_envs = 3
    n_steps_per_env = 4

    env = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=True)
    memory = OnPolicyReplay()

    observations = env.reset()

    step = 0

    for _ in range(n_steps_per_env):
        actions = env.action_space.sample()
        next_observations, rewards, dones, _ = env.step(actions)
        step += len(observations)
        memory.update(observations, actions, rewards, next_observations, dones)
        observations = next_observations

    assert step == memory.total_experiences

    print(f"num_envs = {num_envs}, n_steps_per_env = {n_steps_per_env}, total timesteps = {step}")
    print()
    pprint(memory.sample())
