# Introdução ao Aprendizado por Reforço - 2020

Este repositório contém os materiais usados em aula no curso de verão **Introdução ao Aprendizado por Reforço** oferecido pelo [Curso de Verão do IME-USP](https://www.ime.usp.br/~verao/index.php) de 11/02/2020 a 16/02/2020.

## Staff
[LIAMF](https://liamf-usp.github.io/liamf-page/): Grupo PAR (Planejamento e Aprendizado por Reforço)

**Professores**: Ângelo Gregório Lovatto (@angelolovatto), Thiago Pereira Bueno (@thiagopbueno)

**Monitor**: Renato Scaroni (@renato-scaroni)

**Coordenadora**: Leliane Nunes de Barros



## Descrição do curso

Introdução aos Processos de Decisão Markovianos; Gradiente de política. Algoritmo REINFORCE e a técnica da score-function; Método actor-critic (A2C); Aprendizado da função valor para redução da variância do gradiente da política. Compromisso entre viés e variância; O curso será desenvolvido utilizando slides e atividades práticas com exercícios de modelagem de problemas e aplicação de métodos aprendidos em problemas benchmark.

**Requisito**: Familiaridade com estatística, probabilidade básicas e cálculo no R^n. Apesar de todos os conceitos necessários serem apresentados durante o curso. 

**Público**: Profissionais da área de IA. Alunos de graduação e pós-graduação interessados na área de aprendizado por reforço. 

---

## Preparação

* [Material Preliminar](Preliminaries/Preliminaries.pdf): Fundamentação matemática e estatística para o acompanhamento do curso
* [Tutorial de configuração do ambiente](Preliminaries/tutorial_ambiente.txt)
* [Material adicional sobre entropia](Preliminaries/entropia.ipynb)

---

## Aula 1 - Introdução / MDPs / OpenAI Gym

![](https://i.imgur.com/mgLaEHt.png)

**Objetivos**:
- Familiarizar-se com os objetivos e formato do curso
- Ter uma ideia geral sobre possíveis aplicações de RL
- Aprender os conceitos básicos e vocabulário de RL
- Entender as diferenças entre RL e Supervised Learning (SL)

**Materiais**:
* [Slides](Aula%201/Aula%201%20-%20Introdução.pdf)
* [Notebook](Aula%201/Aula-1/Aula%201%20-%20Parte%20Prática%20-%20Agentes%20&%20Ambientes.ipynb
)
* [Solução](Aula%201/Solução/Aula%201%20-%20Parte%20Prática%20-%20Agentes%20&%20Ambientes.ipynb
)

## Aula 2 - Policy Gradients / Política Estocástica / TensorFlow + Keras

![](https://i.imgur.com/bALMWJb.png)


**Objetivos**:
* Entender a abordagem de otimização de políticas como busca no espaço de parâmetros da política
* Implementar um primeiro agente baseado no algoritmo REINFORCE
* Familiarizar-se com a API básica de construção de modelos (i.e., redes neurais) em Keras
* Familiarizar-se com métodos de Deep Learning usando TensorFlow 2.X

**Materiais**:
* [Slides](Aula%202/Aula%202%20-%20Policy%20Gradient.pdf
)
* [Notebook](Aula%202/Aula-2/Aula%202%20-%20Parte%20Prática%20-%20Policy%20Gradients.ipynb
)
* [Solução](Aula%202/Solução/Aula%202%20-%20Parte%20Prática%20-%20Policy%20Gradients.ipynb
)

## Aula 3 - Função Valor e Redução de Variância / Baselines

![](https://i.imgur.com/OvYjrt3.png)


**Objetivos**:
* Relacionar as propriedades do estimador REINFORCE com a performance do agente
* Verificar experimentalmente o efeito de redução de variância do estimador de Policy Gradient calculado com reward-to-go
* Incorporar a função Valor como baseline para os retornos das trajetórias no REINFORCE
* Familiarizar-se com o aprendizado de função Valor via regressão sobre os retornos das trajetórias

**Materiais**:
* [Slides](Aula%203/Aula%203%20-%20Redução%20de%20Variância.pdf
)
* [Notebook](Aula%203/Aula-3/Aula%203%20-%20Parte%20Prática%20-%20Redução%20de%20Variância%20e%20Função%20Valor.ipynb
)

## Aula 4 - Actor-Critic (A2C) / Generalized Advantage Estimation (GAE)

![](https://i.imgur.com/lmi3PaE.png)

**Objetivos**:
* Familiarizar-se com os componentes Actor e Critic
* Entender o papel da função Valor na estimativa truncada dos retornos
* Ter um primeiro contato com truques de implementação tipicamente utilizados e RL

**Materiais**:
* [Slides](Aula%204/Aula%204%20-%20Actor-Critic.pdf
)
* [Notebook](Aula%204/Aula-4/Aula%204%20-%20Parte%20Prática.ipynb
)
* [Solução](Aula%204/Solução/Aula%204%20-%20Parte%20Prática.ipynb
)

## Aula 5 - Tópicos Avançados: Desafios de RL

![](https://i.imgur.com/tVanrBj.png)

**Objetivos**:
* Entender algumas das limitações e dificuldades fundamentais de Deep RL
* Familiarizar-se com técnicas avançadas de algoritmos Actor-Critic
* Ter uma visão geral sobre diferentes áreas de pesquisa em RL

**Materiais**:
* [Slides](Aula%205/Aula%205%20-%20Desafios%20de%20RL.pdf
)

---

## Referências

### Livros

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) (Sutton & Barto 2018, 2nd Edition)
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow, Bengio and Courville, 2016)


### Frameworks e bibliotecas

* [OpenAI Gym](http://gym.openai.com/)
* [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
* [TensorFlow Probability](https://www.tensorflow.org/probability)
* [Keras](https://keras.io/)


### Blogs, sites e outros recursos na web

* [OpenAI Spinning Up: Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [Intuitive RL (Reinforcement Learning): Introduction to Advantage-Actor-Critic (A2C)]()
* [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)


### Vídeos

* [Neural Networks & Backpropagation (3Blue1Brown)](https://www.3blue1brown.com/neural-networks)
* [MIT 6.S091: Introduction to Deep Reinforcement Learning (Deep RL)](https://www.youtube.com/watch?v=zR11FLZ-O9M)
* [CS 285: Deep RL, Decision Making, and Control: Policy Gradients](https://www.youtube.com/watch?v=Ds1trXd6pos)
* [CS 285: Deep RL, Decision Making, and Control: Actor-Critic](https://www.youtube.com/watch?v=EKqxumCuAAY)
* [Deep RL Bootcamp Lecture 4A: Policy Gradients:](https://www.youtube.com/watch?v=S_gwYj1Q-44)
* [Deep RL Bootcamp Lecture 4B: Policy Gradients Revisited](https://www.youtube.com/watch?v=tqrcjHuNdmQ)
* [OpenAI Baselines: ACKTR & A2C](https://openai.com/blog/baselines-acktr-a2c/)


### Artigos Científicos: 

* [Challenges of Real-World Reinforcement Learning](https://arxiv.org/abs/1904.12901) (Dulac-Arnold, Mankowitz, and Hester, 2019)
* [Reinforcement Learning Applications](https://arxiv.org/abs/1908.06973) (Li, 2019)
* [RECSIM: A Configurable Simulation Platform for Recommender Systems](http://www.cs.toronto.edu/~cebly/Papers/Article_RecSim_arXiv_1909.04847.pdf) (Ie, Eugene, et al., 2019)
* [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) (Sutton, R.S., McAllester, D.A., Singh, S.P. and Mansour, Y., 2000) 
* [Deep Learning in Neural Networks: An Overview](https://arxiv.org/abs/1404.7828) (Schmidhuber, 2014)
* [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) (Ruder, 2017)
