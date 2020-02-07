import abc

class RLAgent:
    """
    Classe abstrata que define a interface básica de um agente RL.

    Args:
        obs_space:     especificação do espaço de observações do ambiente.
        action_space:  especificação do espaço de ações do ambiente.
        config (dict): (opcional) configurações de hiper-parâmetros.
    """
    
    __metaclass__ = abc.ABCMeta

    def __init__(self, obs_space, action_space, config=None):
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config

    @abc.abstractmethod
    def act(self, obs):
        """
        Escolhe uma ação para ser tomada dada uma observação do ambiente.
        
        Args: 
            obs: observação do ambiente.
        
        Return:
            action: ação válida dentro do espaço de ações.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self, obs, action, reward, next_obs, done):
        """
        Registra na memória do agente uma transição do ambiente.

        Args:
            obs:            observação do ambiente antes da execução da ação.
            action:         ação escolhida pelo agente.
            reward (float): escalar indicando a recompensa obtida após a execução da ação.
            next_obs:       nova observação recebida do ambiente após a execução da ação.
            done (bool):    True se a nova observação corresponde a um estado terminal, False caso contrário.

        Return:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        """
        Método de treinamento do agente. A partir das experiências de sua memória,
        o agente aprende um novo comportamento.

        Args: 
            None

        Return:
            None
        """     
        raise NotImplementedError