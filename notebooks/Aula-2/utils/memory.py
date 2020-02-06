class OnPolicyReplay:
    
    def __init__(self):
        self.data_keys = ["states", "actions", "rewards", "next_states", "dones"]
        self.total_experiences = 0
        self.n_episodes = 0
        self.reset()
        
    def reset(self):
        for key in self.data_keys:
            setattr(self, key, [])

        self.current_episode = {key: [] for key in self.data_keys}
        self.batch_size = 0
    
    def update(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        
        for i, key in enumerate(self.data_keys):
            self.current_episode[key].append(transition[i])
            
        if done:
            self.n_episodes += 1
            for key in self.data_keys:
                getattr(self, key).append(self.current_episode[key])

            self.current_episode = {key: [] for key in self.data_keys}
        
        self.total_experiences += 1
        self.batch_size += 1
    
    def sample(self):
        batch = {key: getattr(self, key) for key in self.data_keys}
        self.reset()
        return batch
    
    
if __name__ == "__main__":
    import gym
    import numpy as np
    
    env = gym.make("MountainCarContinuous-v0")
    memory = OnPolicyReplay()

    total_experiences = 0
    n_episodes = 0
    total_rewards = []

    for _ in range(5):
        n_episodes += 1
        total_reward = 0.0
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            memory.update(obs, action, reward, next_obs, done)
            total_experiences += 1
            total_reward += reward
            next_obs = obs
            if done:
                total_rewards.append(total_reward)
                break

    assert memory.total_experiences == total_experiences
    assert memory.batch_size == total_experiences
    batch = memory.sample()
    assert all(len(batch[key]) == n_episodes for key in memory.data_keys)
    assert memory.total_experiences == total_experiences
    assert memory.batch_size == 0
