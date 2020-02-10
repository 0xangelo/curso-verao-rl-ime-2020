import gym
import numpy as np


header = """

 __       __                      _______   __              _______             __                    __     
|  \     /  \                    |       \ |  \            |       \           |  \                  |  \    
| $$\   /  $$  ______            | $$$$$$$\| $$            | $$$$$$$\  ______  | $$____    ______   _| $$_   
| $$$\ /  $$$ /      \           | $$__| $$| $$            | $$__| $$ /      \ | $$    \  /      \ |   $$ \  
| $$$$\  $$$$|  $$$$$$\          | $$    $$| $$            | $$    $$|  $$$$$$\| $$$$$$$\|  $$$$$$\ \$$$$$$  
| $$\$$ $$ $$| $$    $$          | $$$$$$$\| $$            | $$$$$$$\| $$  | $$| $$  | $$| $$  | $$  | $$ __ 
| $$ \$$$| $$| $$$$$$$$ __       | $$  | $$| $$_____       | $$  | $$| $$__/ $$| $$__/ $$| $$__/ $$  | $$|  
\
| $$  \$ | $$ \$$     \|  \      | $$  | $$| $$     \      | $$  | $$ \$$    $$| $$    $$ \$$    $$   \$$$$
 \$$      \$$  \$$$$$$$| $$       \$$   \$$ \$$$$$$$$       \$$   \$$  \$$$$$$  \$$$$$$$   \$$$$$$     \$$$$ 
                        \$                                                                                   

"""
print(header)
print("Sinta-se como um robô em um problema de aprendizado por reforço !")
print()

print("Tecle Enter para começar ...")
input()


env = gym.make("FrozenLake-v0")
actions = ', '.join([str(i) for i in range(env.action_space.n)])

n_episodes = 10
for episode in range(n_episodes):    
    t = 0
    print("########### Início de episódio {episode} ###########")
    print()
    
    history = []

    obs = env.reset()
    print(f">> observação[t=0] = {obs}")
    
    history.append(f"obs={obs}")
    
    total_reward = 0.0
    
    done = False
    while not done:

        action = int(input(f">> Escolha uma ação [{actions}] : "))
        next_obs, reward, done, _ = env.step(action)
        reward = np.random.uniform(-1.0, 1.0) * np.random.normal(loc=reward, scale=10.0)

        total_reward += reward

        history.append(f"a={action}")
        history.append(f"r={reward:.2f}")
        history.append(f"obs={next_obs}")
        
        t += 1
        print()
        print(f">> observação[t={t}] = {next_obs}")
        print(f">> recompensa[t={t}] = {reward:.2f}")
        print()
        
        print(f">> histórico = [{', '.join(history)}]")
    
    print()
    print(f"########### Fim de episódio {episode}: retorno = {total_reward:.2f} ###########")
    print()
